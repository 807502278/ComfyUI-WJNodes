class AnyType(str):
    def __init__(self, _):
        self.is_any_type = True

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")

CATEGORY_NAME = "WJNode/Other"


class any_data:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "data_array": ("LIST",),
                "data_1": (any,),
                "data_2": (any,),
                "data_3": (any,),
                "data_4": (any,),
                "data_5": (any,),
                "data_6": (any,),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("LIST", any, any, any, any, any, any,)
    RETURN_NAMES = ("data_array", "data_1", "data_2",
                    "data_3", "data_4", "data_5", "data_6",)
    FUNCTION = "any_data_array"

    def any_data_array(self, data_array=[None, None, None, None, None, None],
                       data_1=None,
                       data_2=None,
                       data_3=None,
                       data_4=None,
                       data_5=None,
                       data_6=None):

        if data_1 is None:
            data_1 = data_array[0]
        else:
            data_array[0] = data_1

        if data_2 is None:
            data_2 = data_array[1]
        else:
            data_array[1] = data_2

        if data_3 is None:
            data_3 = data_array[2]
        else:
            data_array[2] = data_3

        if data_4 is None:
            data_4 = data_array[3]
        else:
            data_array[3] = data_4

        if data_5 is None:
            data_5 = data_array[4]
        else:
            data_array[4] = data_5

        if data_6 is None:
            data_6 = data_array[5]
        else:
            data_array[5] = data_6

        return (data_array, data_1, data_2, data_3, data_4, data_5, data_6,)


class show_type:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": (any,),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TypeName",)
    OUTPUT_NODE = True
    FUNCTION = "TypeName"

    def TypeName(self, data, ):
        name = str(type(data).__name__)
        print(f"Prompt:The input data type is --->{name}")
        return (name,)


NODE_CLASS_MAPPINGS = {
    "any_data": any_data,
    "show_type": show_type,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "any_data": "any data",
    "show_type": "show type",
}
