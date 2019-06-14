# Sets
# Structure


# Sets are groups of things without order. Sets are defined
#	using parenthesis, brackets, and the word 'set'. Sets 
#	can contain any type of object, but in general all 
#	objects in a set are of the same type. A set cannot
#	however contain data structures (lists, tuples, or
#	other sets).

print "Ex. 1:", set(["This", "is", "a", "set"])
print "Ex. 2:", set([2, 3, 4])
print "Ex. 3:", set([True, False])
# Empty set
print "Ex. 4:", set([])
# Errors
#print "Error:", set([set([1, 2]), set([3, 4]), set([5, 6])])
#print "Error:", set([["a", "a", "a"], ["b", "c", "d"]])
print

# Sets differ from lists in that the user cannot specify
#	the order in which elements appear, and only one instance
#	of an element can be in any given set.

print "Ex. 5:", set([1, 3, 5, 2, 4])
print "Ex. 6:", set(['a', 'c', 'b'])
print "Ex. 7:", set([3, 6, 1, 7, 7, 7, 7, 7, 2])
print "Ex. 8:", set(['a', 'x', 'e', 'x', 'y', 'x'])

# Unlike lists, sets cannot be referenced by index.

s = set(['a', 'b', 'c', 'd', 'e'])
#print "Error: ", s[0]
#print "Error: ", s[1:2]
#s[2] = "z"

# You can still use len() to find the number of elements
#	in a list.

s = set([1, 2, 3, 4])
print "Ex. 9:", len(s)
print

# Here are the mutator methods for sets:

# set.add(item) adds the given item to the set. Note that
#	this is not necessarily at the beginning or end of the
#	set.

things = set(["hi", "hello", "greetings"])
things.add("bye")
print "Ex. 10:", things
things = set([])
things.add(2)
things.add(1)
things.add(6)
things.add(3)
print "Ex. 11:", things
print

# set.remove(item) removes the item from the set if it is
#	present, but throws an error if it is not in the set.

things = set(['a', 'b', 'c', 'd'])
things.remove('b')
print "Ex. 12:", things
#things.remove('x')
print

# set.pop() removes an arbitrary item from the set.
#	Returns an error if the set is empty.

things = set(['b', 'a', 'e', 'c', 'd'])
things.pop()
print "Ex. 13:", things
things.pop()
things.pop()
print "Ex. 14:", things
things.pop()
things.pop()
print "Ex. 15:", things
#things.pop()
print

# set.union(iterable) returns a set containing all of the
#	elements from both set and iterable.
# set.intersection(iterable) returns a set containing all
#	of the elements that both set and iterable have in common.
# Note: iterables include lists, tuples, and sets.

s1 = set([1, 2, 3, 4, 5])
s2 = set([4, 5, 6, 7, 8])
l = [1, 3, 5, 7]
t = (4, 8)
print "Ex. 16:", s1.union(s2)
print "Ex. 17:", s1.intersection(s2)
print "Ex. 18:", s1.union(l)
print "Ex. 19:", s1.intersection(l)
print "Ex. 20:", s1.union(t)
print "Ex. 21:", s1.intersection(t)
print

# Another useful method is set.copy(). This returns an exact
#	replica of the list, which is useful for when you want to
#	modify a set in a loop.

s1 = set(['a', 'b', 'c', 'd', 'e'])
s2 = s1.copy()
print "Ex. 22:", s1
print "Ex. 23:", s2
print

# Be careful when assigning two variables to be the same set.
#	They can become 'aliases' of the same set, causing a
#	change to one of them to affect both of them. This is the
#	same behavior as lists.

set_a = set([1, 2, 3])
set_b = set_a
set_b.add(10)
print "Ex. 24:", set_a, set_b
# This is fine. a and b are completely separate lists.
set_a = set([1, 2, 3])
set_b = set([1, 2, 3])
set_b.add(10)
print "Ex. 25:", set_a, set_b
# This is also fine.
set_a = set([1, 2, 3])
set_b = set_a.copy()
set_b.add(10)
print "Ex. 26:", set_a, set_b
print


