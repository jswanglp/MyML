# coding: utf-8
# 以命令行参数的形式创建相关对象，并设置任何属性,继承自dict类
class Bunch(dict):  
    def __init__(self,*args,**kwds):  
        super(Bunch,self).__init__(*args,**kwds)  
        self.__dict__ = self
        
        
# exp:
# >> x=Bunch(age="54",address="Beijing")
# >> x.age 
# >> 54