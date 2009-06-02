greeting :: String -> String
greeting name = 
  "Hi, " ++ name ++ "! Your name is " ++ charcnt ++ " chars long."
  where charcnt = show(length(name))

main = do
  putStrLn "Happy days! What's your name? "
  input <- getLine
  putStrLn $ greeting(input)

