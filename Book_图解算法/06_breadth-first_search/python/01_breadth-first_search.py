from collections import deque

def person_is_seller(name):
      return name[-1] == 'm'

graph = {}
graph["you"] = ["alice", "bob", "claire"]
graph["bob"] = ["anuj", "peggy"]
graph["alice"] = ["peggy"]
graph["claire"] = ["thom", "jonny"]
graph["anuj"] = []
graph["peggy"] = []
graph["thom"] = []
graph["jonny"] = []

def search(name):
    search_queue = deque()
    search_queue += [name]
    # This is how you keep track of which people you've searched before.
    searched = set()
    i = 0
    print(f"{i} = {search_queue}")
    while search_queue:
        i += 1
        person = search_queue.popleft()
        # Only search this person if you haven't already searched them.
        if person in searched:
            continue
        if person_is_seller(person):
            print(person + " is a mango seller!")
            return True
        search_queue += graph[person]
        print(f"{i}, search_queue = {search_queue}")
        # Marks this person as searched
        searched.add(person)
        print(f"{i}, searched = {searched}\n")
    return False

search("you")

# name = "you"
# search_queue = deque()
# search_queue += [name]
# # This is how you keep track of which people you've searched before.
# searched = set()
# while search_queue:
#     person = search_queue.popleft()
#     # Only search this person if you haven't already searched them.
#     if person in searched:
#         continue
#     if person_is_seller(person):
#         print(person + " is a mango seller!")
#         # return True
#     search_queue += graph[person]
#     # Marks this person as searched
#     searched.add(person)
