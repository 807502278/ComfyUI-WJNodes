from ..moduel.str_edit import str_edit
from ..moduel.list_edit import random_select


CATEGORY_NAME = "WJNode/Prompt"

class Random_Select_Prompt:
    DESCRIPTION = """
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Prompt":("STRING",{"default":"","multiline": True}),
                "select_number":("INT",{"default":1,"min":1,"max":4096,"step":1}),
                "Original_data_deduplication":("BOOLEAN",{"default":False}),
                "allow_duplicates":("BOOLEAN",{"default":False}),
                "keep_order":("BOOLEAN",{"default":True}),
                "random_seed":("INT",{"default":1,"min":0,"max":99999999,"step":1})
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("Prompt","Prompt_list")
    FUNCTION = "random_tag"
    def random_tag(self,Prompt,select_number,Original_data_deduplication,allow_duplicates,keep_order,random_seed):
        #输入检查
        str_list = []
        if isinstance(Prompt,str):
            Prompt = Prompt.replace(', ', ',')
            Prompt = Prompt.replace('，', ',')
            Prompt = Prompt.replace('， ', ',')
            str_list = Prompt.split(",")
        elif isinstance(Prompt,list) and isinstance(Prompt[0],str):
            str_list = Prompt
        elif isinstance(Prompt,(tuple,set)) and isinstance(Prompt[0],str):
            str_list = Prompt.tolist()
        else :
            raise TypeError("Error:Prompt data type error, can only input string or string list !")
        #...
        if Original_data_deduplication:
            str_list = list(set(str_list))
        str_list = random_select(str_list, select_number, random_seed, allow_duplicates, keep_order)
        return (str_edit.list_to_str(str_list),str_list)



NODE_CLASS_MAPPINGS = {
    #WJNode/Prompt
    "Random_Select_Prompt": Random_Select_Prompt,
    
}