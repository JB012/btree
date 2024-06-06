from __future__ import annotations
import json
from typing import List

def ceiling(a: int, b: int):
    return -(a // -b)

def getValue(environment: List[(int, str)], key: int):
    for elem in environment:
        if key == elem[0]:
            return elem[1]
    return None

env = []

class Node():
    def  __init__(self,
                  keys     : List[int]  = None,
                  values   : List[str] = None,
                  children : List[Node] = None,
                  parent   : Node = None):
        self.keys     = keys
        self.values   = values
        self.children = children
        self.parent   = parent

class Btree():
    def  __init__(self,
                  m    : int  = None,
                  root : Node = None):
        self.m    = m
        self.root = root

    def dump(self) -> str:
        def _to_dict(node) -> dict:
            return {
                "keys": node.keys,
                "values": node.values,
                "children": [(_to_dict(child) if child is not None else None) for child in node.children]
            }
        if self.root == None:
            dict_repr = {}
        else:
            dict_repr = _to_dict(self.root)
        return json.dumps(dict_repr,indent=2)

    def insert(self, key: int, value: str):
        if self.root == None:
            env.append((key,value))
            self.root = Node (keys=[key], values=[value], children=[None, None], parent=None)
        else:
            self.traverse_insert(self.root, key, value, 0)

    def inorder_successor(self, node:Node, key: int):
        count = 0
        found = 0
        for child in node.children:
            for eachKey in child.keys:
                if key < eachKey:
                    found = 1
                    break
            if found == 1:
                break
            count += 1
        
        rightChild = node.children[count]
        leftChild = rightChild.children[0]

        if leftChild:
            while (leftChild.children[0] != None):
                leftChild = leftChild.children[0]
            return (leftChild, leftChild.keys[0])
        else: return (rightChild, rightChild.keys[0])

    def underfull_cases (self, root:Node, index:int):
        temp = [None]*((len(root.keys)))

        for i in range(0, len(root.keys)):
            temp[i] = root.keys[i]

        if root.parent:
            if (index != 0 and root.parent.children[index-1] != None):
                if (len(root.parent.children[index-1].keys) > (ceiling(self.m, 2) - 1)):
                    totalKeys = len(root.keys) + len(root.parent.children[index-1].keys)
                    while(len(root.keys) < totalKeys//2):
                        root.parent.children[index-1] = self.right_rotate(root.parent.children[index-1], index-1)


            if (root.keys == temp):
                if (root.parent):
                    if (index != len(root.parent.children)-1 and root.parent.children[index+1] != None):
                        if (len(root.parent.children[index+1].keys) > (ceiling(self.m, 2) - 1)):
                            totalKeys = len(root.keys) + len(root.parent.children[index+1].keys)
                            while(len(root.keys) < totalKeys//2):
                                root.parent.children[index+1] = self.left_rotate(root.parent.children[index+1], index+1)
            if root.keys == temp:
                if root.parent.children[index-1] and index - 1 != -1:
                    if len(root.keys) + len(root.parent.children[index-1].keys) + 1 <= (self.m - 1):
                        copy = root
                        root = self.merge(root, root.parent.children[index-1], index, "left")

                        if root.parent:
                            if copy in root.parent.children:
                                root.parent.children.remove(copy)
                            
                       

                        
                if root.keys == temp:
                    if root.parent.children[index+1] and index + 1 != len(root.parent.children):
                        if len(root.keys) + len(root.parent.children[index+1].keys) + 1 <= (self.m - 1):
                            copy = root
                            root = self.merge(root, root.parent.children[index+1], index, "right")
                            
                            if root.parent:
                                if copy in root.parent.children:
                                    root.parent.children.remove(copy)
                                    
        return root
    
    def traverse_delete_search (self, root:Node, key:int):
        if key not in root.keys:
            #if not in root then check children
            for child in root.children:
                if child:
                    res = self.traverse_delete_search(child, key) 
                    if key in res.keys:
                        return res
        return root

    def traverse_delete (self, root:Node, key: int, value: str, index: int):
       
        root = self.traverse_delete_search(root, key)
    
        if root.parent:
            index = root.parent.children.index(root)
        else: 
            #Root
            index = 0

        isLeaf = 1

        for child in root.children:
            if child != None:
                isLeaf = 0
    
        root.keys.remove(key)
        root.keys.sort()
        root.values = self.sortValue(root.keys)
        if isLeaf == 1:
            root = self.restruct_children(root)
        else:
            replacementKey = self.inorder_successor(root, key)[1]
            leafNode = self.inorder_successor(root, key)[0]
           

            root.keys.append(replacementKey)
            root.keys.sort()
           
            root.values = self.sortValue(root.keys)


            leafNode.keys.remove(replacementKey)
            leafNode.keys.sort()
            leafNode.values = self.sortValue(leafNode.keys)
            leafNode = self.restruct_children(leafNode)

            if len(leafNode.keys) == (ceiling(self.m, 2)-2):
                count = 0
                for child in leafNode.parent.children:
                    if leafNode == child:
                        break
                    count+=1
                
                leafNode = self.underfull_cases(leafNode, count)
              
                if leafNode.parent == None:
                    root = leafNode

        env.remove((key,value))

        if isLeaf == 1 and len(root.keys) == (ceiling(self.m,2) - 2):
            root = self.underfull_cases(root, index)
        
   
    def traverse_insert (self, root:Node, key: int, value: str, index: int):
        count = 0
        for eachKey in root.keys:
            if key < eachKey:
                break
            count += 1

        if count < len(root.children) and root.children[count]:
            self.traverse_insert(root.children[count], key, value, count)
        else:
            #At the leaf
            root.keys.append(key)
            env.append((key,value))
            root.keys.sort()
            root.values = self.sortValue(root.keys)
            root.children += [None]
        
            if self.m == len(root.keys):
                temp = [None]*((len(root.keys)))

                #Left rotation
                for i in range(0, len(root.keys)):
                    temp[i] = root.keys[i]
    
                if (root.parent):
                    if (index != 0 and root.parent.children[index-1] != None):
                        totalKeys = len(root.keys) + len(root.parent.children[index-1].keys)
                        while(len(root.keys) > ceiling(totalKeys, 2)):
                            root = self.left_rotate(root, index)

                if (root.keys == temp):
                    if (root.parent):
                        if (index != len(root.parent.children)-1 and root.parent.children[index+1] != None):
                            totalKeys = len(root.keys) + len(root.parent.children[index+1].keys)
                            while(len(root.keys) > ceiling(totalKeys, 2)):
                                root = self.right_rotate(root, index)

                if(root.keys == temp):
                    root = self.split(root)


    def right_rotate (self, root: Node, count: int):
        if (root.parent.children[count+1] != None):
            if (len(root.parent.children[count+1].keys) < self.m-1):
                largestKey = root.keys[-1]
            
                largestChild = root.children[-1]

                for eachKey in root.parent.keys:
                    if eachKey > largestKey:
                        parentKey = eachKey
                        break
                

                root.parent.children[count+1].keys.append(parentKey)
                root.parent.children[count+1].keys.sort()

                if largestChild:
                    largestChild.parent = root.parent.children[count+1]
                root.parent.children[count+1].children.append(largestChild)
                root.parent.children[count+1] = self.restruct_children(root.parent.children[count+1])
                root.parent.children[count+1].values = self.sortValue(root.parent.children[count+1].keys)

                root.parent.keys.remove(parentKey)

                root.parent.keys.append(largestKey)
                root.parent.keys.sort()
                root.parent.values = self.sortValue(root.parent.keys)

                root.keys.remove(largestKey)
                root.values = self.sortValue(root.keys)
                root.children.remove(largestChild)
                root = self.restruct_children(root)
        return root

    def left_rotate (self, root: Node, count: int):
        if (root.parent):
            if (root.parent.children[count-1] != None and count != 0):
                if (len(root.parent.children[count-1].keys) < self.m-1):
                    smallestKey = root.keys[0]
                    smallestChild = root.children[0]

                    for eachKey in root.parent.keys:
                        if eachKey < smallestKey:
                            parentKey = eachKey

                    root.parent.children[count-1].keys.append(parentKey)
                    root.parent.children[count-1].keys.sort()

                    if smallestChild:
                        smallestChild.parent = root.parent.children[count-1]
                    root.parent.children[count-1].children.append(smallestChild)
                    root.parent.children[count-1] = self.restruct_children(root.parent.children[count-1])
                    root.parent.children[count-1].values = self.sortValue(root.parent.children[count-1].keys)

                    root.parent.keys.remove(parentKey)

                    root.parent.keys.append(smallestKey)
                    root.parent.keys.sort()
                    root.children.remove(smallestChild)
                    root.parent.values = self.sortValue(root.parent.keys)

                    root.keys.remove(smallestKey)
                    root.values = self.sortValue(root.keys)
                    root = self.restruct_children(root)
                    
        return root
    

    def merge(self, node:Node, sibling:Node, index:int, side: str):
        keyDemoted = None
       
        if side == "left":
            if len(node.keys) == 0:
                siblingKey = sibling.keys[0]
                for eachKey in node.parent.keys:
                    if siblingKey < eachKey:
                        keyDemoted = eachKey
                        break

            else:
                for eachKey in node.parent.keys:
                    if node.keys[0] > eachKey:
                        keyDemoted = eachKey
        elif side == "right":
            if len(node.keys) == 0:
                siblingKey = sibling.keys[0]
                for eachKey in node.parent.keys:
                    if eachKey < siblingKey:
                        keyDemoted = eachKey
            else:
                for eachKey in node.parent.keys:
                    if node.keys[-1] < eachKey:
                        keyDemoted = eachKey
                        break
                
            
        sibling.keys.append(keyDemoted)
        sibling.keys += node.keys
        sibling.keys.sort()
        sibling.values = self.sortValue(sibling.keys)

        sibling.children += node.children
        sibling = self.restruct_children(sibling)

        node.parent.keys.remove(keyDemoted)
        node.parent.keys.sort()
        node.parent.values = self.sortValue(node.parent.keys)
        
        node = sibling
        if len(node.parent.keys) <= ceiling(self.m, 2) - 2 and node.parent.parent:
            grandParent = node.parent.parent
            node.parent = self.underfull_cases(node.parent, grandParent.children.index(node.parent))
            for child in node.parent.children:
                child.parent = node.parent
        #Root can't be underfull but if there's an empty root then we promote merged node
        elif len(node.parent.keys) == 0 and node.parent.parent == None:
                node.parent = None
                self.root = node                       
                            
        return node     


    
    def split (self, node: Node):
        keyPromoted = 0
        newNode = None
        if (self.m % 2 == 1):
            keyPromoted = self.m // 2
        else: keyPromoted = (self.m // 2) - 1

        if (node.parent == None):
            newNode = Node (keys=[node.keys[keyPromoted]], values=[], children=[None, None], parent=None)
            newNode.values.append(getValue(env, node.keys[keyPromoted]))
        else:
            node.parent.keys.append(node.keys[keyPromoted])
            node.parent.keys.sort()
            node.parent.values = self.sortValue(node.parent.keys)
            node.parent.children += [None]

        newLeftNode = Node (keys = node.keys[:keyPromoted], values=[], children=node.children[:keyPromoted+1], parent=None)
        newLeftNode.values = self.sortValue(newLeftNode.keys)

        newRightNode = Node (keys = node.keys[keyPromoted+1:], values=[], children=node.children[keyPromoted+1:], parent=None)
        newRightNode.values = self.sortValue(newRightNode.keys)

        if (node.parent != None):
            newLeftNode.parent = node.parent
            newRightNode.parent = node.parent
        else:
            newLeftNode.parent = newNode
            newRightNode.parent = newNode

        #Update parents of children
        if (newLeftNode.children):
            for child in newLeftNode.children:
                if child:
                    child.parent = newLeftNode
            
        if (newRightNode.children):
            for child in newRightNode.children:
                if child:
                    child.parent = newRightNode

        
        if (ceiling(self.m, 2) <= len(newLeftNode.children) <= self.m and ceiling(self.m, 2) <= len(newRightNode.children) <= self.m):
            if (node.parent != None):
                node.parent.children.append(newLeftNode)
                node.parent.children.append(newRightNode)
            else:
                newNode.children[0] = newLeftNode
                newNode.children[1] = newRightNode
        
        
        if (node.parent != None):
            
            node.parent.children.remove(node)
            node.parent = self.restruct_children(node.parent)

            if (len(node.parent.keys) == self.m):
                #Finding where the node is and if it has siblings around it.
                index = 0
                hasGrandParent = 0
                if node.parent.parent:
                    hasGrandParent = 1
                    for child in node.parent.parent.children:
                        if node.parent.keys == child.keys:
                            break
                        index += 1

                temp = [None] * (len(node.parent.keys))
                for i in range(0, len(node.parent.keys)):
                    temp[i] = node.parent.keys[i]
                    
                if hasGrandParent:
                    
                    if index >= 1:
                        if node.parent.children[index-1]:
                            totalKeys = len(node.parent.keys) + len(node.parent.parent.children[index-1].keys)
                            while (len(node.parent.keys) > ceiling(totalKeys, 2)):
                                node.parent = self.left_rotate(node.parent, index)

                                if temp == node.parent.keys:
                                    break
                              
                    
                    if temp == node.parent.keys:
                        if index != len(node.parent.parent.keys):
                            if node.parent.children[index+1]:
             
                                totalKeys = len(node.parent.keys) + len(node.parent.parent.children[index+1].keys)

                                while (len(node.parent.keys) > ceiling(totalKeys, 2)):
                                    node.parent = self.right_rotate(node.parent, index)

                                    if temp == node.parent.keys:
                                        break
                                  
                if temp == node.parent.keys:    
                    node.parent = self.split(node.parent)
        else: 
            self.root = newNode

        return node

    def sortValue(self, keyList:List[int]):
        newValueList = [None]*(len(keyList))
        index = 0
        for key in keyList:
            newValueList[index] = getValue(env, key)
            index += 1
        return newValueList

    #Get the smallest key smaller than key
    def getSmallestKey(self, key: int, node: Node):
        smallest = 1000000
        for child in node.children:
            if child:
                if len(child.keys) > 0:
                    if child.keys[0] < smallest and child.keys[0] > key:
                        smallest = child.keys[0]
        return smallest

    def restruct_helper(self, index:int, node:Node, key:int, childrenLst: List[Node]):
        for child in node.children:
                if child:
                    if len(child.keys) > 0:
                        if child.keys[0] == key:
                            if index != len(childrenLst):
                                childrenLst[index] = child
        return childrenLst

    def restruct_children(self, node: Node):
        smallestKey = 100000
        for child in node.children:
            if child:
                if len(child.keys) > 0:
                    if child.keys[0] < smallestKey:
                        smallestKey = child.keys[0]
        newChildren = [None] * (len(node.keys) + 1)

        
        newChildren = self.restruct_helper(0, node, smallestKey, newChildren)


        for i in range(1, len(node.children)):
            if i < len(newChildren):
                smallestKey = self.getSmallestKey(smallestKey, node)
                newChildren = self.restruct_helper(i, node, smallestKey, newChildren)

        node.children = newChildren

        return node


    def delete(self, key: int):
        self.traverse_delete(self.root, key, getValue(env, key), 0)
        

    def search_helper(self, root: Node, key: int, lst:List[str]):
        count = 0
        if root:
            for eachKey in root.keys:
                if key == eachKey:
                    lst.append(getValue(env, key))
                    return lst
                if key > eachKey:
                    count += 1
            
            lst.append(count)
            self.search_helper(root.children[count], key, lst)
        return lst

    def search(self,key) -> str:
        value_list = self.search_helper(self.root, key, [])
        return json.dumps(value_list)
