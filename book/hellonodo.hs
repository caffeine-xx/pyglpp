import System.IO

main =
  putStrLn "Good Morning.  WTF Are You? " >>
  getLine >>=
  (\inpStr -> putStrLn $ "Okay, " ++ inpStr ++ ", welcome I guess.")

