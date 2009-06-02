import System.IO
import Data.Char(toUpper)

main = do
  input <- readFile "input.txt"
  writeFile "output.txt" (map toUpper input)

