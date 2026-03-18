# Python大型项目中用到的python内容
## 实例
实例是类的具体化
```Python
//定义类
class Dog:
    # 初始化方法：每次创建新实例时自动调用
    def __init__(self, name, age):
        self.name = name  # 实例属性：每只狗的名字不同
        self.age = age    # 实例属性：每只狗的年龄不同
    
    def bark(self):
        # 实例方法：描述这个实例能做什么
        return f"{self.name} 说：汪汪！"
//创建实例
# 创建第一个实例
dog1 = Dog("旺财", 3)
# 创建第二个实例
dog2 = Dog("小白", 5)
//使用实例
print(dog1.name)       # 输出: 旺财
print(dog2.name)       # 输出: 小白
print(dog1.bark())     # 输出: 旺财 说：汪汪！
print(dog2.bark())     # 输出: 小白 说：汪汪！
# 修改一个实例的属性，不会影响另一个
dog1.age = 4
print(dog1.age)        # 输出: 4
print(dog2.age)        # 输出: 5 (保持不变)
```
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
## @Property 装饰器
它的主要作用是将类的方法转换为只读属性，或者为属性的获取、设置和删除提供自定义逻辑（Getter/Setter/Deleter），同时保持调用时的简洁语法（像访问普通变量一样）
例子：
```Python
class Person:
    def __init__(self, name):
        self._name = name  # 内部变量通常加下划线
    
    @property
    def name(self):
        """这是 getter 方法，现在可以像属性一样访问"""
        print("正在获取 name...")
        return self._name

# 使用
p = Person("Alice")
print(p.name)  # 输出: 正在获取 name... \n Alice
# p.name = "Bob"  # ❌ 报错！AttributeError: can't set attribute (因为是只读的)
```
## @dataclass 装饰器
**核心作用是自动化生成样板代码**。在引入它之前，如果你要写一个主要用来存储数据的类（类似 Java 的 POJO 或 C 的结构体），你需要手动写 __init__、__repr__（生成友好的字符串表示）、__eq__（实现对象相等性比较） 等方法，非常繁琐。@dataclass 可以根据你的类型注解，自动帮你生成这些方法。
例子：
```Python
传统写法
class Person:
    def __init__(self, name: str, age: int, job: str = "Unknown"):
        self.name = name
        self.age = age
        self.job = job

    def __repr__(self):
        return f"Person(name={self.name!r}, age={self.age!r}, job={self.job!r})"

    def __eq__(self, other):
        if not isinstance(other, Person):
            return False
        return (self.name, self.age, self.job) == (other.name, other.age, other.job)

# 实例化
p = Person("Alice", 30)
print(p)  # Person(name='Alice', age=30, job='Unknown')
```

```Python
使用dataclass
from dataclasses import dataclass
@dataclass
class Person:
    name: str
    age: int
    job: str = "Unknown"  # 支持默认值

# 实例化（完全一样）
p = Person("Alice", 30)
print(p)  # 自动生成了 __repr__: Person(name='Alice', age=30, job='Unknown')

# 自动支持比较
p2 = Person("Alice", 30, "Unknown")
print(p == p2)  # True (自动生成了 __eq__)
```

# GIL 锁对多线程的影响
Python 的 GIL (Global Interpreter Lock，全局解释器锁) 是 CPython 解释器（最常用的 Python 实现）中的一个互斥锁。它的核心作用是确保同一时刻只有一个线程在执行 Python 字节码。
这一机制对多线程编程产生了深远且复杂的影响，主要体现在 “CPU 密集型” 和 “I/O 密集型” 两种截然不同的场景下。
- A. CPU 密集型任务（瓶颈所在）
现象：当你有多个线程在进行大量数学计算（如矩阵运算、图像处理、复杂逻辑循环）时。
机制：线程 A 获取 GIL，开始执行字节码。线程 B 想要执行，但必须等待 A 释放 GIL。CPython 通常每隔一定时间（或执行一定数量的字节码指令）强制线程释放 GIL，以便其他线程运行。
结果：即使在 8 核 CPU 上，同一时刻也只有 1 个核在跑 Python 代码，其他 7 个核在空转等待。
额外开销：频繁的 GIL 争夺和线程上下文切换（保存/恢复寄存器、栈等）会消耗 CPU 资源，导致总耗时可能比单线程还长。
- B. I/O 密集型任务（优势所在）
现象：当线程在进行网络请求、读写文件、查询数据库时。
机制：线程 A 发起一个 I/O 操作（如 requests.get()）。在等待服务器响应的这段时间里，线程 A 主动释放 GIL。线程 B 立即获取 GIL，开始处理自己的任务（可能是另一个网络请求）。当 A 的 I/O 完成，它重新尝试获取 GIL 继续执行。
结果：虽然同一时刻只有一个线程在跑 Python 代码，但 CPU 没有浪费在“等待 I/O”上，而是充分利用起来处理其他线程的逻辑。这实现了高效的并发。