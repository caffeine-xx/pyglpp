module GlobRegex (
  matchesGlob,
  globToRegex
  ) where

import Text.Regex.Posix ((=~))

globToRegex cs   = "^" ++ globToRegex' cs ++ "$"

globToRegex' :: String -> String
globToRegex' glob = case glob of
  ('*':cs) -> ".*" ++ globToRegex' cs
  ('?':cs) -> "." ++ globToRegex' cs
  ("[!"++c:cs) -> "[^" ++ charClass cs
  ('[':c:cs) -> "[" ++ charClass cs
  ('[':_) -> error "unterminated char class"
  (c:cs) -> escape c ++ globToRegex' cs

escape :: Char -> String
escape c | c `elem` regexChars = '\\' : c
         | otherwise = c
  where regexChars = "\\+^$.(){}|]"

charClass :: String -> String
charClass str = case str of
  (']':cs)  = 
