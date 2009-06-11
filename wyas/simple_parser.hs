import Text.ParserCombinators.Parsec hiding (spaces)
import System.IO
import System.Environment
import Monad(liftM)
import Control.Monad.Error

data LispVal = Atom String
  | List [LispVal]
  | DottedList [LispVal] LispVal
  | Number Integer
  | String String
  | Bool Bool

data LispError = NumArgs Integer [LispVal]
  | TypeMismatch String LispVal
  | Parser ParseError
  | BadSpecialForm String LispVal
  | NotFunction String String
  | UnboundVar String String
  | Default String

instance Show LispVal where show = showVal
instance Show LispError where show = showError
instance Error LispError where
  noMsg = Default "An error occurred"
  strMsg = Default

type ThrowsError = Either LispError

main :: IO ()
main = do
  args <- getArgs
  evald <- return $ liftM show $ readExpr (args !! 0) >>= eval
  putStrLn $ extractValue $ trapError evald

showVal :: LispVal -> String
showVal v = case v of
  (String s) -> "\"" ++ s ++ "\""
  (Atom n) -> n
  (Number n) -> show n
  (Bool True) -> "#t"
  (Bool False) -> "#f"
  (List l) -> "(" ++ unwordsList l ++ ")"
  (DottedList head tail) -> "(" ++ unwordsList head ++ " . " ++ showVal tail ++ ")"

-- parser
 
readExpr :: String -> ThrowsError LispVal
readExpr input = case parse parseExpr "lisp" input of
  Left err  -> throwError $ Parser err
  Right val -> return val

unwordsList :: [LispVal] -> String
unwordsList = unwords . map showVal

parseExpr :: Parser LispVal
parseExpr = parseAtom
  <|> parseString
  <|> parseNumber
  <|> parseQuoted
  <|> parseQuasiQuoted 
  <|> do 
    char '('
    x <- try parseList <|> parseDottedList
    char ')'
    return x

symbol :: Parser Char
symbol = oneOf "!#$%&|*+-/:<=>?@^_~"

spaces :: Parser ()
spaces = skipMany1 space

parseAtom :: Parser LispVal
parseAtom = do 
  first <- letter <|> symbol
  rest <- many (letter <|> digit <|> symbol)
  let atom = first:rest
  return $ case atom of
    "#t" -> Bool True
    "#f" -> Bool False
    otherwise -> Atom atom

parseString :: Parser LispVal
parseString = do 
  char '"'
  x <- many (escaped <|> noneOf "\"")
  -- x <- many (noneOf "\"")
  char '"'
  return $ String x

escaped :: Parser Char
escaped = do
  char '\\'
  x <- oneOf ['"', 'n', 'r', 't', '\\']
  return $ case x of
    'n' -> '\n'
    'r' -> '\r'
    't' -> '\t'
    '\\' -> '\\'
    '"' -> '\"'

parseNumber :: Parser LispVal
parseNumber = liftM (Number . read) $ many1 digit

parseNumber2 :: Parser LispVal
parseNumber2 = do
  num <- many1 digit
  return $ (Number . read) num

parseNumber3 :: Parser LispVal
parseNumber3 = (many1 digit) >>= \v -> (return . Number . read) v 


parseList :: Parser LispVal
parseList = liftM List $ sepBy parseExpr spaces

parseDottedList :: Parser LispVal
parseDottedList = do
  head <- endBy parseExpr spaces
  tail <- char '.' >> spaces >> parseExpr
  return $ DottedList head tail

parseQuoted :: Parser LispVal
parseQuoted = do
  char '\''
  x <- parseExpr
  return $ List [Atom "quote", x]

parseQuasiQuoted :: Parser LispVal
parseQuasiQuoted = do
  char '`'
  x <- parseExpr
  return $ List [Atom "quasiquote", x]

-- evaluator

eval :: LispVal -> ThrowsError LispVal
eval val = case val of
  val@(String _) -> return val
  val@(Number _) -> return val
  val@(Bool _) -> return val
  (List [Atom "quote", val]) -> return val
  (List [Atom "if", pred, conseq, alt]) -> do
    result <- eval pred
    case result of
      Bool False -> eval alt
      Bool True -> eval conseq
      otherwise -> throwError $ TypeMismatch "Conditional needs bools" result
  (List (Atom func : args)) -> mapM eval args >>= apply func
  _ -> throwError $ BadSpecialForm "Unrecognized special form" val


apply :: String -> [LispVal] -> ThrowsError LispVal
apply func args = maybe (throwError $ NotFunction "Unrecognized primitive" func)
  ($ args) 
  (lookup func primitives)

-- primitive operations

primitives :: [(String, [LispVal] -> ThrowsError LispVal)]
primitives = [
  ("+",numericBinop (+)),
  ("-",numericBinop (-)),
  ("*",numericBinop (*)),
  ("/",numericBinop div),
  ("mod", numericBinop mod),
  ("quotient", numericBinop quot),
  ("remainder", numericBinop rem),
  ("symbol?",  is_atom),
  ("boolean?", is_bool),
  ("=", numBoolBinop (==)),
  ("<", numBoolBinop (<)),
  (">",  numBoolBinop (>)),
  ("/=", numBoolBinop (/=)),
  (">=", numBoolBinop (>=)),
  ("<=", numBoolBinop (<=)),
  ("&&", boolBoolBinop (&&)),
  ("||", boolBoolBinop (||)),
  ("string=?", strBoolBinop (==)),
  ("string<?", strBoolBinop (<)),
  ("string>?", strBoolBinop (>)),
  ("string<=?", strBoolBinop (<=)),
  ("string>=?", strBoolBinop (>=)),
  ("car", car), 
  ("cons", cons),
  ("cdr", cdr),
  ("eq?", eqv),
  ("eqv?", eqv),
  ("equal?", equal)]

-- primitive: numerical

numericBinop :: (Integer -> Integer -> Integer) -> [LispVal] -> ThrowsError LispVal
numericBinop op single@[_] = throwError $ NumArgs 2 single
numericBinop op parms = mapM unpackNum parms >>= return . Number . foldl1 op

unpackNum :: LispVal -> ThrowsError Integer
unpackNum (Number n) = return n
unpackNum (String n) = 
  let parsed = reads n in
    if null parsed
      then throwError $ TypeMismatch "number" $ String n
      else return $ fst $ parsed !! 0
unpackNum (List [n]) = unpackNum n
unpackNum wrong = throwError $ TypeMismatch "number" wrong

-- primitive: type checking

is_atom :: [LispVal] -> ThrowsError LispVal
is_atom [Atom _] = return $ Bool True
is_atom arg = false_or_one arg

is_bool :: [LispVal] -> ThrowsError LispVal
is_bool [Bool _] = return $ Bool True
is_bool arg = false_or_one arg

false_or_one :: [LispVal] -> ThrowsError LispVal
false_or_one  [_] = return $ Bool False
false_or_one args = throwError $ NumArgs 1 args

-- primitive: binary ops

boolBinop :: (LispVal -> ThrowsError a) -> (a -> a -> Bool) -> [LispVal] -> ThrowsError LispVal
boolBinop unpacker op args = 
  if length args /= 2
    then throwError $ NumArgs 2 args
    else do 
          left <- unpacker $ args !! 0
          right <- unpacker $ args !! 1
          return $ Bool $ left `op` right

numBoolBinop = boolBinop unpackNum
strBoolBinop = boolBinop unpackStr
boolBoolBinop = boolBinop unpackBool

unpackStr :: LispVal -> ThrowsError String
unpackStr v = case v of
  String s -> return s
  Number s -> return $ show s
  Bool   s -> return $ show s
  other    -> throwError $ TypeMismatch "string" v

unpackBool :: LispVal -> ThrowsError Bool
unpackBool (Bool b) = return b
unpackBool        x = throwError $ TypeMismatch "bool" x

-- primitive: list

car :: [LispVal] -> ThrowsError LispVal
car [list] = case list of
  (List (x:xs)) -> return x
  (DottedList (x:xs) y) -> return x
  _ -> throwError $ TypeMismatch "list" list
car x = throwError $ NumArgs 1 x

cdr :: [LispVal] -> ThrowsError LispVal
cdr [list] = case list of
  (List (_:xs)) -> return $ List xs
  (DottedList [x] y) -> return y
  (DottedList (_:xs) y) -> return $ DottedList xs y
  _ -> throwError $ TypeMismatch "list" list
cdr bad = throwError $ NumArgs 1 bad

cons :: [LispVal] -> ThrowsError LispVal
cons pair = case pair of
  [x, List []] -> return $ List [x]
  [x, List xs] -> return $ List (x:xs)
  [x, DottedList xs xl] -> return $ DottedList (x:xs) xl
  [x, y] -> return $ DottedList [x] y
  bad -> throwError $ NumArgs 2 pair

eqv :: [LispVal] -> ThrowsError LispVal
eqv [x, y] = case (x, y) of 
  (Bool l1, Bool l2)     -> return $ Bool $ l1 == l2
  (Atom l1, Atom l2)     -> return $ Bool $ l1 == l2
  (String l1, String l2) -> return $ Bool $ l1 == l2
  (Number l1, Number l2) -> return $ Bool $ l1 == l2
  (List _, List _)       -> listEq eqv [x, y]
  (DottedList _ _, DottedList _ _) -> listEq eqv [x, y]
  other                  -> return $ Bool $ False
eqv bad = throwError $ NumArgs 2 bad

equal :: [LispVal] -> ThrowsError LispVal
equal [x, y] = case (x, y) of
  (List _, List _)       -> listEq equal [x, y]
  (DottedList _ _, DottedList _ _) -> listEq equal [x, y]
  (a1, a2) -> do 
    primEq <- liftM or $ mapM (unpackEquals a1 a2) 
      [AnyUnpacker unpackNum, AnyUnpacker unpackStr, AnyUnpacker unpackBool]
    eqvEq  <- eqv [a1, a2]
    return $ Bool $ (primEq || let (Bool x) = eqvEq in x)
equal bad = throwError $ NumArgs 2 bad

listEq :: ([LispVal] -> ThrowsError LispVal) -> [LispVal] -> ThrowsError LispVal
listEq comp [x, y] = case (x, y) of
  (DottedList xs xe, DottedList ys ye) -> listEq comp [List $ xs ++ [xe], List $ ys ++ [ye]] 
  (List xs, List ys) -> return $ Bool $ (length xs == length ys) &&
    (and $ map eqvPair $ zip xs ys) where 
      eqvPair (x1, x2) = case eqv [x1, x2] of
        Left err -> False -- should never be executed, because only NumArgs errors
        Right (Bool val) -> val
  (p, q) -> return $ Bool False

data Unpacker = forall a. Eq a => AnyUnpacker (LispVal -> ThrowsError a)

unpackEquals :: LispVal -> LispVal -> Unpacker -> ThrowsError Bool
unpackEquals a1 a2 (AnyUnpacker unpacker) = 
  do up1 <- unpacker a1
     up2 <- unpacker a2
     return $ up1 == up2
  `catchError` (const $ return False)
-- errors
--

showError :: LispError -> String
showError err = case err of
  (UnboundVar msg var) -> msg ++ ": " ++ var
  (BadSpecialForm msg frm) -> msg ++ ": " ++ show frm
  (NotFunction msg fn) -> msg ++ ": " ++ show fn
  (NumArgs num found) -> "# args expected: " ++ show num ++
    ", found: " ++ unwordsList found
  (TypeMismatch exp found) -> "type error, expected: " ++ show exp ++
    ", found: " ++ show found
  (Parser parseErr) -> "Parse error at: " ++ show parseErr

trapError action = catchError action (return . show)

extractValue :: ThrowsError a -> a
extractValue (Right val) = val

