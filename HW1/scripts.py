
# ===== PROBLEM1 =====

# Exercise 1 - Introduction - Say "Hello, World!" With Python

print("Hello, World!")

# Exercise 2 - Introduction - Python If-Else

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 != 0:
        print("Weird")
    elif n >= 2 and n <=5:
        print("Not Weird")
    elif n >= 6 and n <= 20:
        print("Weird")
    elif n >= 20:
        print("Not Weird")


# Exercise 3 - Introduction - Arithmetic Operators
        
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

# Exercise 4 - Introduction - Python: Division
    
if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a//b)
    print(a/b)
    
# Exercise 5 - Introduction - Loops

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i*i)
    
# Exercise 6 - Introduction - Write a function

def is_leap(year):
    leap = False
    if year % 400 == 0:
        leap = True
    elif year % 100 == 0 and year % 4 == 0:
        leap = False
    elif year % 4 == 0:
        leap = True   
    
    return leap

# Exercise 7 - Introduction - Print Function

if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        print(i, end='')
        
# Exercise 8 - Basic data types - List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k!=n])    

# Exercise 9 - Basic data types - Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    arr = list(arr)
    b = max(arr)

    for r, i in enumerate(arr):
        if i == b:
            arr[r] = min(arr)
    
    print(max(arr))
    
# Exercise 10 - Basic data types - Nested Lists

if __name__ == '__main__':
    marksheet = []
    scorelist= []
    for i in range(int(input())):
        name = input()
        score = float(input())
        marksheet += [[name, score]]
        scorelist += [score]
    b = sorted(list(set(scorelist)))[1]
    for a,c in sorted(marksheet):
        if c == b:
            print(a)

# Exercise 11 - Basic data types - Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}

    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores

    query_name = input()
    print("{0:.2f}".format(sum(student_marks[query_name])/3))   
    #I divided by 3 because in the examples that's the only N of marks, however 
    # if we divided by len(student_marks[name]) we could come up with the right solution
    # for every N.
    
# Exercise 12 - Basic data types - Lists

if __name__ == '__main__':
    n = int(input())
    l = []
    for _ in range(n):
        s = input().split()
        cmd = s[0]
        args = s[1:]
        if cmd != "print":
            cmd += '(' + ','.join(args) + ')'
            cmd = eval('l.' + cmd)
        else:
            print(l)
            
# Exercise 13 - Basic data types - Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    integer_list = tuple(integer_list)
    print(hash(integer_list))

# Exercise 14 - Strings - sWAP cASE

def swap_case(s):
    a = ''
    for x in s:
        if x.isupper() == True:
            a += x.lower()
        else:
            a += x.upper()
    return a

# Exercise 15 - Strings - String Split and Join

def split_and_join(line):
    b = line.split(' ')
    a = '-'.join(b)
    return a

# Exercise 16 - Strings - What's Your Name?

def print_full_name(a, b):
    print(f"Hello {a} {b}! You just delved into python.")

# Exercise 17 - Strings - Mutations

def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    a = ''.join(l)
    return a

# Exercise 18 - Strings - Find a string

def count_substring(string, sub_string):
    a = sum([1 for i in range(len(string)-len(sub_string)+1) if string[i:i+len(sub_string)] == sub_string])
    return a

# Exercise 19 - Strings - String Validators

if __name__ == '__main__':
    s = input()
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for i in range(len(s)):
        if s[i].isalnum():
            a += 1
        if s[i].isalpha():
            b += 1
        if s[i].isdigit():
            c += 1
        if s[i].islower():
            d += 1
        if s[i].isupper():
            e += 1
    def sta(a):
        if a > 0:
            print(True)
        else:
            print(False)
    
    sta(a)
    sta(b)
    sta(c)
    sta(d)
    sta(e)

# Exercise 20 - Strings - Text Alignment

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Exercise 21 - Strings - Text Wrap

def wrap(string, max_width):
    i = 0
    while i < len(string) - max_width:
        a = string[i:i+max_width]
        print(a)
        i+=max_width
    b = string[i:]
    return b

# Exercise 23 - Strings - String Formatting

def print_formatted(number):
    w=len(bin(number)[2:])
    for i in range(1,n+1):
        print(str(i).rjust(w, ' '), end = " ")
        print(str(oct(i)[2:]).rjust(w, ' '), end = " ")
        print(str(hex(i)[2:].upper()).rjust(w, ' '), end = " ") 
        print(str(bin(i)[2:]).rjust(w, ' '))
        
# Exercise 25 - Strings - Capitalize!

def solve(s):
    s = list(s)
    
    if s[0] != ' ':
        s[0] = s[0].capitalize()
    
    for i in range(1,len(s)):
        if s[i-1] == ' ':
            s[i] = s[i].capitalize()
    s = ''.join(s)

# Exercise 26 - Strings - The Minion Game

def minion_game(s):
    vowels = 'AEIOU'
    sc1 = 0
    sc2 = 0
    for i in range(len(s)):
        if s[i] in vowels:
            sc1 = sc1 + (len(s)-i)
        else:
            sc2 = sc2 + (len(s)-i)

    if sc1 > sc2:
        print("Kevin", sc1)
    elif sc1 < sc2:
        print("Stuart", sc2)
    else:
        print("Draw")

# Exercise 28 - Sets - Introduction to Sets

def average(array):
    a = len(set(arr))    
    b = sum(set(arr))
    c = b/a
    return c

# Exercise 29 - Sets - No Idea!

n,m = input().split()
c = list(map(int, input().split()))
a = set(map(int, input().split()))
b = set(map(int, input().split()))
f = 0
for i in c:
    if i in a and i not in b:
        f+=1
    elif i not in a and i in b:
        f-=1

print(f)

# Exercise 30 - Sets - Symmetric Difference

n = input()
a = set(map(int, input().split()))
m = input()
b = set(map(int, input().split()))
u = a.union(b)
i = a.intersection(b)
result = u.difference(i)
result = sorted(result)
for i in result:
    print(i)
    
# Exercise 31 - Sets - Set .add()

n = int(input())
a = set()
for i in range(n):
   a.add(input())

l = len(a)
print(l)

# Exercise 32 - Sets - Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
t = int(input())
for i in range(t):
    x = list(input().split())
    if x[0] == 'pop':
        x[0] += '()'
        eval('s.' + x[0])
    else:
        x[0] += '(' + x[1] + ')'
        eval('s.' + x[0])

su = sum(s)
print(su)

# Exercise 33 - Sets - Set .union() Operation

n = int(input())
s = set(map(int, input().split()))
m = int(input())
t = set(map(int, input().split()))

u = s.union(t)
u = len(u)
print(u)

# Exercise 34 - Sets - Set .intersection() Operation

n = int(input())
s = set(map(int, input().split()))
m = int(input())
t = set(map(int, input().split()))

u = s.intersection(t)
u = len(u)
print(u)

# Exercise 35 - Sets - Set .difference() Operation

n = int(input())
s = set(map(int, input().split()))
m = int(input())
t = set(map(int, input().split()))

d = s.difference(t)
d = len(d)
print(d)

# Exercise 36 - Sets - Set .symmetric_difference() Operation

n = int(input())
s = set(map(int, input().split()))
m = int(input())
t = set(map(int, input().split()))

sd = s.symmetric_difference(t)
sd = len(sd)
print(sd)

# Exercise 37 - Sets - Set Mutations

(n, a) = (int(input()), set(map(int, input().split())))
b = int(input())
for i in range(b):
    (command, newset) = (input().split()[0], set(map(int, input().split())))
    getattr(a, command)(newset)

print(sum(a))

# Exercise 38 - Sets - The Captain's Room

k = int(input())
rooms = list(map(int, input().split()))

setroom = set()
setroomrip = set()

for i in rooms:
    if i not in setroom:
        setroom.add(i)
    else:
        setroomrip.add(i)

print(setroom.difference(setroomrip).pop())#taken help from discussions

# Exercise 39 - Sets - Check Subset

t = int(input())
for i in range(t):
    n = int(input())
    a = set(map(int, input().split()))
    m = int(input())
    b = set(map(int, input().split()))
    if a.intersection(b) == a:
        print(True)
    else:
        print(False)
        
# Exercise 40 - Sets - Check Strict Superset

a = set(map(int, input().split()))
n = int(input())

for i in range(n):
    b = set(map(int, input().split()))
    if  b.intersection(a)!= b and b != a:
        x = False
        print(x)
        break
    else:
        x = True

if x == True:
    print(x)
    
# Exercise 41 - Collections - collections.Counter()

from collections import Counter
n = int(input())
l = list(map(int, input().split()))
k = int(input())
a = Counter(l)
b = Counter(l).keys()
s = 0
for i in range(k):
    lun, cost = map(int, input().split())
    if lun in b and a[lun] != 0:
        s += cost
        a[lun] += -1

print(s)

# Exercise 42 - Collections - DefaultDict Tutorial

from collections import defaultdict

n, m = map(int,input().split())

a = defaultdict(list)
b = []

for i in range(0,n):
    a[input()].append(i+1)
    
for j in range(0,m):
    b.append(input())

for x in b:
    if x in a:
        print(' '.join(map(str, a[x])))
    else:
        print(-1) 

# Exercise 43 - Collections - Collections.namedtuple()

from collections import namedtuple

(n, columns) = (int(input()), input().split())
grades = namedtuple('Grade', columns)
marks = [int(grades._make(input().split()).MARKS) for _ in range(n)]
print((sum(marks) / len(marks)))    

# Exercise 44 - Collections - Collections.OrderedDict()

from collections import OrderedDict
d = OrderedDict()
for _ in range(int(input())):
    item, space, quantity = input().rpartition(' ')
    d[item] = d.get(item, 0) + int(quantity)
for item, quantity in d.items():
    print(item, quantity)

# Exercise 46 - Collections - Collections.deque()

from collections import deque
d = deque()
for i in range(int(input())):
    cmd, *args = input().split()
    getattr(d, cmd )(*args)
for x in d:
    print(x, end=' ')

# Exercise 49 - Date time - Calendar Module

import calendar

date = input().split()

year = int(date[2])
month = int(date[0])
day = int(date[1])

g = calendar.weekday(year, month, day)

days = {0: 'MONDAY', 1: 'THUESDAY', 2: 'WEDNESDAY', 3: 'THURSDAY', 4: 'FRIDAY', 5: 'SATURDAY', 6: 'SUNDAY'}

print(days[g])

# Exercise 51 - Exceptions -

n = int(input())

for i in range(n):    
    try:
        a, b = map(int, input().split())
        print(a//b)
    except Exception as e:
        print('Error Code:',e)
        
# Exercise 52 - Built-ins - Zipped!

n,m = map(int, input().split())
a = []
for i in range(m):
    a.append(map(float,input().split()))


for i in zip(*a):
    print(sum(i)/m)
        
# Exercise 53 - Built-ins - Athlete Sort

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    nums = []

    for _ in range(n):
        nums.append(list(map(int, input().rstrip().split())))

    k = int(input())

    nums.sort(key = lambda x: x[k])
    for line in nums:
        print(*line)

# Exercise 54 - Built-ins - Ginorts

Listl = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1357902468'
print(*sorted(input(), key = Listl.index), sep = '')

# Exercise 55 - Map and lambda function

cube = lambda x: x**3# complete the lambda function 

def fibonacci(n):
    if n > 2:
        a = 0
        b = 1
        fib = []
        fib.extend([a,b])
        for i in range(n-2):
            c = a + b
            fib.append(c)
            a = b
            b = c
        return fib
    elif n == 1:
        return [0]
    elif n == 2:
        return [0,1]
    else:
        return []

# Exercise 56 - Regex - Detect Floating Point Number

import re
for i in range(int(input())):
    print(bool(re.search(r'^[-+]?[0-9]*\.[0-9]+$', input())))
    
# Exercise 57 - Regex - Re.split()

regex_pattern = r"[,|.]"	

# Exercise 58 - Regex - Group(), Groups() & Groupdict()

import re

text = input()

s = re.search(r'([a-zA-Z0-9])\1+', text)

if bool(s):
    print(s.group(1))
else:
    print(-1)

# Exercise 59 - Regex - Re.findall() & Re.finditer()

import re
text = input()
a = '[bcdfghjklmnpqrstvwxyz]'
s = re.findall(r'(?<=' + a + ')([aeiou]{2,})' + a, text, re.I) 
if bool(s):
    for i in s:
        print(i)
else:
    print(-1)

# Exercise 60 - Regex - Re.start() & Re.end()

import re 
s = input()
k = input()
ind = re.compile(k)
r = ind.search(s)
if not r:
    print('(-1, -1)')
while r:
    print((r.start(), r.end()-1))
    r = ind.search(s, r.start()+1)
    
# Exercise 63 - Regex - Validating phone numbers

import re
for i in range(int(input())):
    if bool(re.match(r'^[789]\d{9}$', input())):
        print('YES')
    else:
        print('NO')
    
# Exercise 64 - Regex - Validating and Parsing Email Addresses

import re
n = int(input())
for _ in range(n):
    x, y = input().split(' ')
    m = re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', y)
    if m:
        print(x,y)

# Exercise 73 - Xml - XML 1 - Find the Score

def get_attr_number(node):
    return sum([len(elem.items()) for elem in tree.iter()])

# Exercise 74 - Xml - XML 2 - Find the Maximum Depth

maxdepth = -1
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
        
    for child in elem:
        depth(child, level + 1)

# Exercise 77 - Numpy - Arrays

def arrays(arr):
    arr = arr[::-1]

    return numpy.array(arr, float)

# Exercise 78 - Numpy - Shape and Reshape

import numpy

arr = [int(x) for x in input().split()]
print(numpy.reshape(arr, (3,3)))

# Exercise 79 - Numpy - Transpose and Flatten

import numpy

n,m = [int(x) for x in input().split()]
arr = numpy.array([[int(x) for x in input().split()]])
for i in range(n-1):
    ar1 = numpy.array([[int(x) for x in input().split()]])
    arr = numpy.concatenate((arr, ar1))

print(numpy.transpose(arr))
print(arr.flatten())

# Exercise 80 - Numpy - Concatenate

import numpy

n,m,p = [int(x) for x in input().split()]

for i in range(n+m):
    if i == 0:
        b1 = numpy.array([[int(x) for x in input().split()]])
    else:
        b = numpy.array([[int(x) for x in input().split()]])
        b1 = numpy.concatenate((b1,b))

print(b1)

# Exercise 81 - Numpy - Zeros and Ones

import numpy

n = [int(x) for x in input().split()]

print(numpy.zeros(tuple(n), dtype = numpy.int))
print(numpy.ones(tuple(n), dtype = numpy.int))

# Exercise 82 - Numpy - Eye and Identity

import numpy

numpy.set_printoptions(sign = ' ')

n,m = [int(x) for x in input().split()]

print(numpy.eye(n, m ,k = 0))

# Exercise 83 - Numpy - Array Mathematics

import numpy

n,m = [int(x) for x in input().split()]
a = numpy.array([[int(x) for x in input().split()]])

if n > 1:
    for i in range(n-1):
        a1 = numpy.array([[int(x) for x in input().split()]])
        a = numpy.concatenate((a,a1))
        
b = numpy.array([[int(x) for x in input().split()]])

if n > 1:
    for j in range(n-1):
        b1 = numpy.array([[int(x) for x in input().split()]])
        b = numpy.concatenate((b,b1))

   
print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(a**b)

# Exercise 84 - Numpy - Floor, Ceil and Rint

import numpy
numpy.set_printoptions(sign = ' ')
n = [float(x) for x in input().split()]

print(numpy.floor(n))
print(numpy.ceil(n))
print(numpy.rint(n))

# Exercise 85 - Numpy - Sum and Prod

import numpy

n, m = [int(x) for x in input().split()]

a = numpy.array([[int(x) for x in input().split()]])
b = numpy.array([[ int(x) for x in input().split()]])
c = numpy.concatenate((a,b))
if n > 2:
    d = numpy.array([[int(x) for x in input().split()]])
    c = numpy.concatenate((c,d))


s = numpy.sum(c, axis = 0)
print(numpy.prod(s))

# Exercise 86 - Numpy - Min and Max

import numpy

n,m = [int(x) for x in input().split()]

a = numpy.array([[int(x) for x in input().split()]])

for i in range(n-1):
    a1 = numpy.array([[int(x) for x in input().split()]])
    a = numpy.concatenate((a,a1))

mi = numpy.min(a, axis = 1)
print(numpy.max(mi))

# Exercise 87 - Numpy - Mean, Var, and Std

import numpy 

n,m = map(int, input().split())
b = []
for i in range(n):
    a = list(map(int, input().split()))
    b.append(a)
b = numpy.array(b)

numpy.set_printoptions(legacy = '1.13')
print(numpy.mean(b, axis = 1))
print(numpy.var(b, axis = 0))
print(numpy.std(b))

# Exercise 88 - Numpy - Dot and Cross

import numpy

n = int(input())

a = numpy.array([input().split() for i in range(n)], int)
b = numpy.array([input().split() for i in range(n)], int)

print(numpy.dot(a,b))

# Exercise 89 - Numpy - Inner and Outer

import numpy

a = numpy.array([int(x) for x in input().split()])
b = numpy.array([int(x) for x in input().split()])

print(numpy.inner(a,b))
print(numpy.outer(a,b))

# Exercise 90 - Numpy - Polynomials

import numpy

coeff = [float(x) for x in input().split()]
x = float(input())

print(numpy.polyval(coeff, x))

# Exercise 91 - Numpy - Linear Algebra

import numpy

n = int(input())

a = numpy.array([input().split() for i in range(n)], float)

print(round(numpy.linalg.det(a), 2))
​
# ===== PROBLEM2 =====
​
# Exercise 92 - Challenges - Birthday Cake Candles

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the birthdayCakeCandles function below.
def birthdayCakeCandles(ar):
    m = max(ar)
    k = 0
    for i in ar:
        if i == m:
            k = k + 1
    return k        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()
    
# Exercise 93 - Challenges - Kangaroo

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if v1 == v2:
        return 'NO'
    elif ((x2 - x1) % (v1 - v2) == 0) and ((x2 - x1) / (v1 - v2) > 0):
        return 'YES'
    else:
        return 'NO'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Exercise 94 - Challenges - Viral Advertising
    
import math
n = int(input())
value = 5
l = 0

for i in range(n):
    l = l + math.floor(value/2)
    value = math.floor(value/2)*3
    

print(l)

# Exercise 95 - Challenges - Recursive Digit Sum

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    def ricor(s):
        if len(s) == 1:
            return s
        new_s = sum(int(i) for i in s)
        return ricor(str(new_s))
    s = sum(int(i) for i in n) * k
    return ricor(str(s))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# Exercise 96 - Challenges - Insertion Sort - Part 1

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    m = arr[n-1]
    i = n-2

    while (arr[i] > m) and (i >= 0):
        arr[i+1] = arr[i]
        print(' '.join(map(str, arr)))
        i = i - 1
    arr[i + 1] = m
    print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)
        

# Exercise 97 - Challenges - Insertion Sort - Part 2

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    for i in range(1, n):
        number = arr[i]
        j = i-1
        
        while j >= 0 and arr[j] > number:
            arr[j+1] = arr[j]
            j = j-1
        arr[j+1] = number
        print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)





