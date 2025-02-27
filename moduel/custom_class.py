class AnyType(str):
    def __init__(self, _):
        self.is_any_type = True
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
    
any = AnyType("*")