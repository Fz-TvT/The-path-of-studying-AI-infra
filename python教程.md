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


   