from collections import deque
import numpy as np
import pandas as pd


class MovingAverage:
    def __init__(self,N):
        '''Returns moving average of last N elements added. Removes older elements.'''
        self.N=N
        self.sum = 0
        self.queue = deque()
        
    def append(self,value):
        self.queue.append(value)
        self.sum+=value
        if len(self.queue)>self.N:
            old_value = self.queue.popleft()
            self.sum-=old_value
            
    @property
    def value(self):
        return self.sum/len(self.queue)
        



class Struct(object):
    """
    Structure data type.

    Examples of use:

    A=cStruct()
    A.property1=value1
    A.property2=value2
    ...

    A=cStruct(property1=value1,property2=value2,...)
    """

    def __init__(self,**keywords):
        self._fields=[]
        for sKey in list(keywords.keys()):
            setattr(self,sKey,keywords[sKey])


    def get_str(self,sSeparator):
        sString = ''
        lsAttributes = list(vars(self).keys())

        for sAttribute in lsAttributes:
            if sAttribute[0]!='_':
                attr = getattr(self,sAttribute)
                if isinstance(attr,(np.ndarray, pd.Series, pd.DataFrame)):
                    sString+='%s=...,%s'%(sAttribute,sSeparator)
                else:
                    sAttributeValue = str(getattr(self,sAttribute))
                    if len(sAttributeValue)>30:
                        sString+='%s=...,%s'%(sAttribute,sSeparator)
                    else:
                        sString+='%s=%s,%s'%(sAttribute,sAttributeValue,sSeparator)
        return '{'+sString[:-2]+'}'


    @property
    def pstr(self):
        return self.get_str('\n')

    def __str__(self):
        return self.get_str(' ')

    def __repr__(self):
        return self.get_str(' ')

    def __unicode__(self):
        return self.get_str(' ')

    def __eq__(self, other):
        return self.GetAttributes2ValuesDict()==other.GetAttributes2ValuesDict()

    def __ne__(self, other):
        return self.GetAttributes2ValuesDict()!=other.GetAttributes2ValuesDict()


    def __hash__(self):
        return hash(self.GetAttributes2ValuesTuple())


    def get_defaults(self,oDefaultStruct):
        '''
        Инициализация текущей структуры значениями по умолчанию из другой структуры для тех полей,
        которых нет в текущей структуре, но которые есть в структуре со значениями по умолчанию.
        Пример:
        A=cStruct(i=1,j=2,k=3)
        B=cStruct(a=10,b=20,c=30,i=333)
        A.GetDefaults(B)
        print A
        Structure: a=10 c=30 b=20 i=1 k=3 j=2 '''
        lsDefaultAttributes = list(vars(oDefaultStruct).keys())
        for sDefaultAttribute in lsDefaultAttributes:
            if not hasattr(self,sDefaultAttribute):
                setattr(self,sDefaultAttribute,getattr(oDefaultStruct,sDefaultAttribute))



    @property
    def fields(self):
        return self._fields

    def get_fields(self):
        return self._fields

    @property
    def fields2values(self):
        dAttributes2Values={}
        for sAttribute in list(vars(self).keys()):
            dAttributes2Values[sAttribute]=getattr(self,sAttribute)
        return dAttributes2Values

    def __setattr__(self, name, value):
            super(Struct, self).__setattr__(name, value)
            if name!='_fields':
                if name not in self._fields:
                    self._fields.append(name)

    def __delattr__(self, name):
            super(Struct, self).__delattr__(name)
            if name!='_fields':
                if name in self._fields:
                    self._fields.remove(name)

    
