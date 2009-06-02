main = do
  putStrLn "Name, please: "
  inStr <- getLine
  putStrLn $ "Hello," ++ inStr ++ "! Nice to meet you."
  
