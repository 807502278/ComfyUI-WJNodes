import torch
from ..moduel.custom_class import any


CATEGORY_NAME = "WJNode/compatible" #兼容节点

class image_math_value:
    DESCRIPTION = """
    expression: expression
    clamp: If you want to continue with the next image_math_ralue, 
            it is recommended not to open it
    Explanation: The A channel of the image will be automatically removed, 
            and the shape will be the data shape

    expression:表达式
    clamp:如果要继续进行下一次image_math_value建议不打开
    说明：会自动去掉image的A通道，shape为数据形状
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
            c = c.unsqueeze(-1).repeat(1, 1, 1, 3)
        if d is not None:
            d = d.unsqueeze(-1).repeat(1, 1, 1, 3)

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

    expression:表达式
    clamp:如果要继续进行下一次image_math_value建议不打开
    说明：会自动去掉image的A通道，shape为数据形状
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
            c = c.unsqueeze(-1).repeat(1, 1, 1, 3)
        if d is not None:
            d = d.unsqueeze(-1).repeat(1, 1, 1, 3)

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


class image_math_value_x10:
    DESCRIPTION = """
    expression: Advanced expression
        Function: Perform numerical calculations on images using expressions.The mask will be treated as a 3-channel image.
    Input:
        expression1-3: The expression results corresponding to outputs 1-3.
        RGBA_to_RGB: Convert the input image to a 3-channel image if it is a 4-channel image.
        clamp: Limit the values to 0-1. It is recommended not to open if you want to continue with the next image_math_value.
    Instructions:
        1. The A channel of the image can be optionally removed. shape represents the data shape.
        2. Some torch methods are supported. Please note the output type of the image.
        3. Leave the expressions for the unwanted outputs blank to ignore unnecessary calculations. 

    expression:高级表达式
        功能：使用表达式对图像进行数值计算，mask将被视为3通道图像
    输入：
        expression1-3：对应输出1-3的表达式结果
        RGBA_to_RGB：如果输入image为4通道则将其转为3通道
        clamp:限制数值为0-1，如果要继续进行下一次image_math_value建议不打开
    说明：
        1：可选去掉image的A通道，shape为数据形状
        2：支持部分torch方法，请注意image输出类型
        3：不需要的输出对应的表达式请留空，可忽略不必要的计算
    """
    def __init__(self):
        self.clamp = True
        self.tensors = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression1":("STRING",{"default":"a+b","multiline": True}),
                "expression2":("STRING",{"default":"","multiline": True}),
                "expression3":("STRING",{"default":"","multiline": True}),
                "RGBA_to_RGB":("BOOLEAN",{"default":True}),
                "clamp":("BOOLEAN",{"default":True}),
            },
            "optional": {
                "a":("IMAGE",),"b":("IMAGE",),"c":("IMAGE",),"d":("IMAGE",),
                "e":("MASK",),"f":("MASK",),"g":("MASK",),"h":("MASK",),"i":("MASK",),"j":("MASK",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE","MASK","IMAGE","MASK","IMAGE","MASK")
    RETURN_NAMES = ("image1","mask1","image2","mask2","image3","mask3")
    FUNCTION = "image_math"
    def image_math(self,expression1,expression2,expression3,clamp,RGBA_to_RGB,**kwargs):
        #初始化值
        self.clamp = clamp
        #将mask形状与image对齐方便计算
        try:
            if RGBA_to_RGB: #是否去掉image的A通道
                for k, v in kwargs.items():
                    if v is not None:
                        if v.dim() == 3: self.tensors[k] = v.unsqueeze(-1).repeat(1, 1, 1, 3)
                        elif v.dim() == 4: #去掉image的A通道
                            if v.shape[-1] == 4: self.tensors[k] = v[..., 0:-1]
                            else: self.tensors[k] = v
                        else: print(f"Warning: The input {k} is not standard image data! (wjnodes-image_math_value_x10)")
            else:
                for k, v in kwargs.items():
                    if v is not None:
                        if v.dim() == 3:  self.tensors[k] = v.unsqueeze(-1).repeat(1, 1, 1, 3)
                        elif v.dim() == 4: #遮罩转3通道
                            if v.shape[-1] == 4:  print(f"Warning: The input {k} is 4-channel image data! (wjnodes-image_math_value_x10)")
                            self.tensors[k] = v
                        else: print(f"Warning: The input {k} is not standard image data!")
        except Exception as e1:
            print(e1)
            raise ValueError(f"Error: input error! (wjnodes-image_math_value_x10)")

        return (*self.handle_img(expression1,1),
                *self.handle_img(expression2,2),
                *self.handle_img(expression3,3),
                )
    
    def handle_img(self,expression,n=1):
        mask,image = None,None
        try:
            e_str = [""," "]
            if expression not in e_str:
                image = eval(expression, {}, self.tensors)
        except:
            print(f"Error: Expression{n} error! (wjnodes-image_math_value_x10)")
        if image is not None:
            try:
                if self.clamp: image = torch.clamp(image, 0.0, 1.0)
                if image.dim() == 3: #如果结果是遮罩
                    mask = image
                    image = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                    print("Warning: The result may be a mask. (wjnodes-image_math_value_x10)")
                elif image.dim() == 4: 
                    mask = torch.mean(image, dim=3, keepdim=False)
            except:
                print("Warning: You have calculated non image data! (wjnodes-image_math_value_x10)")
        return (image,mask)



NODE_CLASS_MAPPINGS = {
    "image_math_value":image_math_value,
    "array_count": array_count,
    "any_data": any_data,
    "image_math_value": image_math_value,
    "image_math_value_x10": image_math_value_x10,
}