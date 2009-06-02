import qualified Data.ByteString.Lazy.Char8 as L

closing = readPrice . (!!4) . L.split ','


readPrice :: L.ByteString -> Maybe Int
readPrice str =
  case L.readInt str of 
    Nothing -> Nothing
    Just (dollars, rest) ->
      case L.readInt (L.tail rest) of
        Nothing -> Nothing
        Just (cents, more) ->
          Just (dollars * 100 + cents)

highestClose :: L.ByteString -> Maybe Int
highestClose = maximum . (Nothing:) . map closing . L.lines

highestCloseFrom :: FilePath -> IO ()
highestCloseFrom file = do
  content <- L.readFile file
  print (highestClose content)


