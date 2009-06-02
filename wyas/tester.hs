data MyVal = Atom String 
           | Bool Bool

check :: (Bool -> MyVal) -> MyVal -> Bool
check f (f x) = Bool True
check _ _ = Bool False


