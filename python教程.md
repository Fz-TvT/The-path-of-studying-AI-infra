# Python大型项目中用到的python内容
## 基类的定义和使用
也叫做父类/超类 
**我们首先定义一个基类**

    class Animal:  # 基类
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        print(f"{self.name} makes a sound")
  
  **接着定义子类-继承基类**
  
    class Dog(Animal):  # Dog 继承自 Animal
    def speak(self):  # 重写（override）基类方法
        print(f"{self.name} says Woof!")
    
    def fetch(self):  # 子类新增方法
        print(f"{self.name} is fetching the ball")
  
  **使用**
  
    dog = Dog("Buddy")
    dog.speak()   # 输出: Buddy says Woof! （调用子类重写的方法）
    dog.fetch()   # 输出: Buddy is fetching the ball
    
  **当子类需要调用基类的方法时，应使用super()**
    
    dog = Dog("Buddy")
    dog.speak()   # 输出: Buddy says Woof! （调用子类重写的方法）
    dog.fetch()   # 输出: Buddy is fetching the ball
 
 **抽象基类-强制子类实现特定方法**
 
    from abc import ABC, abstractmethod

    class Shape(ABC):  # 抽象基类
    @abstractmethod # 装饰器 告诉子类方法必须实现 否则子类也会变成抽象类
    def area(self):
        pass # 占位符，让代码语法合理

    class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):  # 必须实现，否则报错
        return 3.14 * self.radius ** 2
        
    ***报错例子***
    class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    # ❌/ 忘记实现 area() 方法
    c = Circle(5)  # TypeError: Can't instantiate abstract class Circle 
               # with abstract methods: area

## Python中的枚举数据类型

枚举的字面含义是指列出有穷集合中的所有元素，即一一列举的意思。在Python中，枚举可以视为是一种数据类型，当一个变量的取值只有几种有限的情况时，我们可以将其声明为枚举类型。例如表示周几的这一变量weekday，只有七种可能的取值，我们就可以将其声明为枚举类型。

那么枚举的类型该如何实现呢？ 我们一个很直观的想法是：可以通过类的方式来实现，变量就是类，变量所有可能的取值作为类的变量。之后访问的时候，通过类名+变量名的方式就可以进行访问
``` Python
class Weekday():
    monday = 1
    tuesday = 2
    wednesday = 3
    thirsday = 4
    friday = 5
    saturday = 6
    sunday = 7
print(Weekday.wednesday)    #  3
``` 
1. 枚举类避免了重复的枚举成员，即避免了重复的键
2. 枚举类使得成员值不能在外部修改
``` Python
from enum import Enum
class Weekday(Enum):
    monday = 1
    tuesday = 2
    wednesday = 3
    thirsday = 4
    friday = 5
    saturday = 6
    sunday = 7
print(Weekday.wednesday)         # Weekday.wednesday      
print(type(Weekday.wednesday))   # <enum 'Weekday'>
print(Weekday.wednesday.name)    # wednesday
print(Weekday.wednesday.value)   # 3
```
**auto() 是 enum 模块提供的自动赋值函数，用于自动生成枚举成员的值，无需手动指定**
auto在不同枚举类型中的行为:
``` Python
Enum（默认从1递增）
from enum import Enum, auto

class Status(Enum):
    PENDING = auto()    # 1
    RUNNING = auto()    # 2
    COMPLETED = auto()  # 3
    FAILED = auto()     # 4
```
``` Python
IntEnum（整数枚举）
from enum import IntEnum, auto

class Priority(IntEnum):
    LOW = auto()    # 1
    MEDIUM = auto() # 2
    HIGH = auto()   # 3

# 可以比较大小
print(Priority.HIGH > Priority.LOW)  # True
```
``` Python
from enum import Flag, auto

class Permission(Flag):
    READ = auto()    # 1  (2^0)
    WRITE = auto()   # 2  (2^1)
    EXECUTE = auto() # 4  (2^2)
    DELETE = auto()  # 8  (2^3)

# 位运算
perm = Permission.READ | Permission.WRITE
print(perm.value)  # 3
```

## 实例方法/类方法/静态方法 三者的差异

|类型	|装饰器|	第一个参数|	访问实例变量	|访问类变量	|调用方式
|-------|-------|-------  |-------     |-------   |------- |
| 实例方法	|无	   |  self	|  ✅	  |    ✅	  | obj.method()
|类方法	 | @classmethod	|  cls	|   ❌ |    ✅  |   Class.method()
|静态方法| @staticmethod| 无 |   ❌|   ❌	|   Class.method()


- 类方法和静态方法可以通过类直接调用，或通过实例直接调用
- 实例方法只能通过实例调用

``` Python
class Example:
    class_var = "类变量" // 类变量是所有实例共享
    
    def __init__(self, value):
        self.instance_var = value // 实例变量（每个实例独立）
    
    # 实例方法
    def instance_method(self):
        return f"{self.instance_var} + {self.class_var}"
    
    # 类方法
    @classmethod
    def class_method(cls):
        return cls.class_var
    
    # 静态方法
    @staticmethod
    def static_method():
        return "静态方法"
    def is_high_salary(salary: float) -> bool:
        return salary > 10000

# 调用
obj = Example("实例变量")
print(obj.instance_method())  # 实例变量 + 类变量
print(Example.class_method()) # 类变量
print(Example.static_method()) # 静态方法
print(obj.is_high_salary(15000))     # True
```