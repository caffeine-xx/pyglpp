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
main :: IO ()
main = getArgs >>= putStrLn . show . eval . readExpr . head

showVal :: LispVal -> String
showVal v = case v of
  (String s) -> "\"" ++ s ++ "\""
  (Atom n) -> n
  (Number n) -> show n
  (Bool True) -> "#t"
  (Bool False) -> "#f"
  (List l) -> "(" ++ unwordsList l ++ ")"
  (DottedList head tail) -> "(" ++ unwordsList head ++ " . " ++ showVal tail ++ ")"

showError :: LispError -> String
showError err = case err of
  (UnboundVar msg var) -> msg ++ ": " ++ var
  (BadSpecialForm msg frm) -> msg ++ ": " ++ show frm
  (NotFunction msg fn) -> msg ++ ": " ++ show fn
  (NumArgs num found) -> "# args expected: " ++ show num ++
    ", found: " ++ unwordsList found
  (TypeMismatch exp found) -> "type error, expected: " ++ show exp ++
    ", found: " ++ show found
  (Parser parseErr) = "Parse error at: " ++ show parseErr

readExpr :: String -> LispVal
readExpr input = case parse parseExpr "lisp" input of
  Left err  -> String $ "no match: " ++ show err
  Right val -> val

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

eval :: LispVal -> LispVal
eval val = case val of
  val@(String _) -> val
  val@(Number _) -> val
  val@(Bool _) -> val
  (List [Atom "quote", val]) -> val
  (List (Atom func : args)) -> apply func $ map eval args
  -- _ -> Bool False

apply :: String -> [LispVal] -> LispVal
apply func args = maybe (Bool False) ($ args) $ lookup func primitives

primitives :: [(String, [LispVal] -> LispVal)]
primitives = [
  ("+",numericBinop (+)),
  ("-",numericBinop (-)),
  ("*",numericBinop (*)),
  ("/",numericBinop div),
  ("mod", numericBinop mod),
  ("quotient", numericBinop quot),
  ("remainder", numericBinop rem),
  ("symbol?",  is_atom),
  ("boolean?", is_bool) ]

numericBinop :: (Integer -> Integer -> Integer) -> [LispVal] -> LispVal
numericBinop op parms = Number $ foldl1 op $ map unpackNum parms

unpackNum :: LispVal -> Integer
unpackNum (Number n) = n
unpackNum (String n) = 
  let parsed = reads n in
    if null parsed
      then 0
      else fst $ parsed !! 0
unpackNum (List [n]) = unpackNum n
unpackNum _ = 0

is_atom :: [LispVal] -> LispVal
is_atom [Atom _] = Bool True
is_atom       _  = Bool False 

is_bool :: [LispVal] -> LispVal
is_bool [Bool _] = Bool True
is_bool       _  = Bool False



