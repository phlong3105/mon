from abc import ABC


class Mixin:

	def __init__(self, b = 1000):
		self.b = b
		print(f"Mixin's init: {b}")

	def method(self):
		print(f"Mixin method 1st call: {self.a}")
		super().method()
		print(f"Mixin method 2nd call: {self.a}")


class A(ABC):

	def __init__(self, a=10):
		print(f"A's init: {a}")
		self.a = a
		self.method()

	def method(self):
		print(f"A's method: {self.a}")
		self.a += 1


class B(A):

	def __init__(self, a=20):
		print(f"B's init: {a}")
		super().__init__()


class C(B):

	def __init__(self, a=30):
		print(f"C's init: {a}")
		super().__init__()


class MyClass(Mixin, C):

	def __init__(self):
		Mixin.__init__(self)
		C.__init__(self)


myclass = MyClass()
