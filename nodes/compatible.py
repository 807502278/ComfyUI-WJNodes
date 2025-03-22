import torch
from ..moduel.custom_class import any


CATEGORY_NAME = "WJNode/compatible" #兼容节点

class array_count:
    DESCRIPTION = """
    Retrieve the shape of array class data and count the number of elements
    获取数组类数据的形状，统计元素数量
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_data": (any,),
                "select_dim":("INT",{"default":0,"min":0,"max":64}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("LIST","INT","INT","INT","INT","INT","INT")
    RETURN_NAMES = ("shape","image-N","image-H","image-W","image-C","sum_count","sel_count",)
    FUNCTION = "element_count"

    def element_count(self, any_data, select_dim):
        n, n1= 1, 1
        s = [0,0,0,0]
        try:
            s = list(any_data.shape)
        except:
            print("Warning: This object does not have a shape property, default output is 0")
        #try:
        shape = list(any_data.shape)
        if len(shape) == 0:
            n, n1= 0, 0
        else:
            for i in range(len(shape)):
                n *= shape[i]
                if i >= select_dim:
                    n1 *= shape[i]
        #except:
        #    print("Error: The input data does not have array characteristics.")
        return (s,*s,n,n1)


class any_data: # 任意数据打组兼容
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

    def any_data_array(self, data_array=None,
                       data_1=None,
                       data_2=None,
                       data_3=None,
                       data_4=None,
                       data_5=None,
                       data_6=None):
        
        if data_array is None:
            data_array=[None, None, None, None, None, None]

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


class image_math_value:
    DESCRIPTION = """
    expression: expression
    clamp: If you want to continue with the next image_math_ralue, 
            it is recommended not to open it
    Explanation: The A channel of the image will be automatically removed, 
            and the shape will be the data shape
    Note: This node has been deprecated, please use image_math-value-v1

    expression:表达式
    clamp:如果要继续进行下一次image_math_value建议不打开
    说明：会自动去掉image的A通道，shape为数据形状
    注意：此节点已被弃用，请使用image_math_value_v1
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression":("STRING",{"default":"a+b","multiline": True}),
                "clamp":("BOOLEAN",{"default":True}),
            },
            "optional": {
                "a":("IMAGE",),
                "b":("IMAGE",),
                "c":("MASK",),
                "d":("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK","LIST")
    RETURN_NAMES = ("image","mask","shape")
    FUNCTION = "image_math"
    def image_math(self,expression,clamp,
                   a=None, b=None, c=None, d=None):
        image = None
        mask = None
        s = 0
        #去掉A通道
        if a is not None:
            if a.shape[-1] == 4: a = a[...,0:-1]
        if b is not None:
            if b.shape[-1] == 4: b = b[...,0:-1]

        #单张批次对齐
        #pass

        #遮罩转3通道
        if c is not None:
            c = c.unsqueeze(-1).expand(-1, -1, -1, 3)
        if d is not None:
            d = d.unsqueeze(-1).expand(-1, -1, -1, 3)

        try:
            local_vars = locals().copy()
            exec(f"image = {expression}", {}, local_vars)
            image = local_vars.get("image")
        except:
            print("Warning: Invalid expression !, will output null value.")

        if image is not None:
            if clamp: image = torch.clamp(image, 0.0, 1.0)
            s = list(image.shape)
            mask = torch.mean(image, dim=3, keepdim=False)
        return (image,mask,s)


NODE_CLASS_MAPPINGS = {
}