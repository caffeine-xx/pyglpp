import System.IO

main = do
  let out = [1..10000000]
  writeFile "big.txt" (show out)

