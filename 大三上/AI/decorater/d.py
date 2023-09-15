def xxx(func):
    def wrapper():
        print("this is in wrapper")
        return func()   # maybe func has something to return
    return wrapper

def bar():
    print("bar")
    return 1

@xxx    # it will pass func in, directly create a new inside function
def dec():
    print("using decorater")
    return 2


bar = xxx(bar)
print(bar())

print(dec())
