#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.11.18

""" 
    Few-shot model based on the BiTT scheme. 
    Reference:  BiTT: Bidirectional Tree Tagging for Joint Extraction of Overlapping Entities and Relations
    Link:       https://arxiv.org/abs/2008.13339
"""

import copy
import json
import math
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

""" Binary tree. """

class BinaryTree(object):
    """ 
        Init a Binary Tree to represent a forest

        Args:

            text:               the text of the entity node
            location:           the location in the sentence of the entity

        Members:

            firstChild:         the first child of the node in the forest
            nextSibling:        the next brother of the node in the forest
            childRelation:      the relation between this node and its child in the forest(for parent)
            siblingRelation:    the relation between this node's parent and its brother in the forest(for parent)
            location:           the location in the sentence of the entity
            parent:             the left child or the right child of parent, and the relation(for child)
    """
    def __init__(self, location):
        super(BinaryTree, self).__init__()
        self.location = location
        self.firstChild = None
        self.nextSibling = None
        self.childRelation = ""
        self.siblingRelation = ""
        self.parent = "root"
    
    def insertChild(self, child_node, child_relation):
        """ 
            Insert a child node of this node in the forest
            Args:
                child_node:         A Binary Tree Node, this node's child
                child_relation:     The parent's label from the relation between child and its parent
            Return:
        """
        if child_relation == str(1): label = str(2)
        elif child_relation == str(2): label = str(1)
        else: label = child_relation

        """ if len(label) == 2:
            if label[1] == str(1): label[1] = str(2)
            elif label[1] == str(2): label[1] = str(1)
            label = label[0] + "-" + label[1]
        else:
            label = child_relation """
        if self.firstChild == None:
            child_node.parent = "left-"+ label
            self.firstChild = child_node
            self.childRelation = child_relation
        else:
            self.firstChild.insertSibling(child_node, child_relation)
    
    def insertSibling(self, sibling_node, sibling_relation):
        """ 
            Insert the next sibling of this node in the forest
            Args:
                sibling_node:       A Binary Tree Node, this node's brother
                sibling_relation:   The sibling's label from the relation between sibling and its parent in the forest
            Return:
        """
        if sibling_relation == str(1): label = str(2)
        elif sibling_relation == str(2): label = str(1)
        else: label = sibling_relation
        
        sibling_node.parent = "right-" + label
        if self.nextSibling == None:
            self.nextSibling = sibling_node
            self.siblingRelation = sibling_relation
        else:
            brother = self.nextSibling
            while brother.nextSibling != None:
                brother = brother.nextSibling
            brother.nextSibling = sibling_node
            brother.siblingRelation = sibling_relation

    def getChildList(self):
        """ 
            Get the child list of this node in the forest
            Return:
                child_list:         The child list
        """
        child_list = []
        if self.firstChild != None:
            child = {}
            child["location"] = self.firstChild.location
            child["label"] = self.childRelation
            child["parent"] = self.firstChild.parent
            child_list.append(child)
            
            left, right = self.firstChild, self.firstChild.nextSibling
            while right != None:
                child = {}
                child["location"] = right.location
                child["label"] = left.siblingRelation
                child["parent"] = right.parent
                child_list.append(child)
                left = right
                right = right.nextSibling        
        return child_list

    def get_tree_dict(self, tokens):
        """ 
            Get the information dictionary of the tree.
            Return:
                tree_dict:          The tree dictionary
        """
        tree_dict = {}
        tree_dict["text"] = " ".join(tokens[self.location[0]:self.location[1]])
        tree_dict["location"] = self.location
        if self.firstChild != None:
            tree_dict["firstChild"] = self.firstChild.get_tree_dict()
            tree_dict["childRelation"] = self.childRelation
        if self.nextSibling != None:
            tree_dict["nextSibling"] = self.nextSibling.get_tree_dict()
            tree_dict["siblingRelation"] = self.siblingRelation
        
        return tree_dict

""" Data Preprocess. """

class BidirectionalTreeTaggingScheme(object):
    """ BiTT Tagging Scheme. """
    def __init__(self, max_seq_len):
        super(BidirectionalTreeTaggingScheme, self).__init__()

        self.max_seq_len = max_seq_len

        self.tag2id = [
            {"NULL": 0, "O": 1, "B": 2, "I": 3, "E": 4, "S": 5},
            {"NULL": 0, "O": 1, "left-1": 2, "left-2": 3, "right-1": 4, "right-2": 5, "root": 6, "right-brother": 7},
            {"NULL": 0, "O": 1, "1": 2, "2": 3, "None": 4},
            {"NULL": 0, "O": 1, "1": 2, "2": 3, "None": 4, "brother": 5}
        ]
        self.id2tag = [
            ["NULL", "O", "B", "I", "E", "S"],
            ["NULL", "O", "left-1", "left-2", "right-1", "right-2", "root", "right-brother"],
            ["NULL", "O", "1", "2", "None"],
            ["NULL", "O", "1", "2", "None", "brother"]
        ]
        self.other_tags, self.begin_tags, self.inner_tags, self.root_tags, self.none_tags = [0, 1], [2, 5], [3, 4], [6, 7], [4]
        self.parts_num = 4
        self.tags_num = [6, 8, 5, 6]
        self.parts_weight = [1.3, 1.0, 1.6, 1.3]
        self.tags_weight = [[] for _ in range(self.parts_num)]
        for i in range(self.parts_num):
            self.tags_weight[i] = [1.0 for j in range(self.tags_num[i])]
            self.tags_weight[i][0] = 0
            self.tags_weight[i][1] = 0.1
    
    def encode_rel_to_bitt_tag(self, sample):
        """ 
            BiTT tags encoder. 
            Args:
                sample:                         {
                                                    "tokens": ["a", "b", "c", ...],
                                                    "relations": [
                                                        {
                                                            "label": "",
                                                            "label_array": ["", "", "", ...],
                                                            "subject": ["a"],
                                                            "object": ["c"],
                                                            "sub_span": (0, 1),
                                                            "obj_span": (2, 3)
                                                        }, ...
                                                    ]
                                                }
            Returns:
                tags_forward + tags_backward:   []
        """
        tokens_len = len(sample["tokens"])
        # Get spots list.
        ent_spots, rel_spots = self.__get_spots_list__(sample)

        # Forward tags.
        ent_spots_forward, rel_spots_forward = copy.deepcopy(ent_spots), copy.deepcopy(rel_spots)
        self.__sort_spots__(ent_spots_forward, reverse=False)
        root_forward = self.__build_binary_tree_fr_spots__(ent_spots_forward, rel_spots_forward)
        tags_forward = [[self.tag2id[0]["O"] for i in range(tokens_len)] for j in range(self.parts_num)]
        self.__get_tags_fr_binary_tree__(root_forward, tags_forward)

        # Backward tags.
        ent_spots_backward, rel_spots_backward = copy.deepcopy(ent_spots), copy.deepcopy(rel_spots)
        self.__sort_spots__(ent_spots_backward, reverse=True)
        root_backward = self.__build_binary_tree_fr_spots__(ent_spots_backward, rel_spots_backward)
        tags_backward = [[self.tag2id[0]["O"] for i in range(tokens_len)] for j in range(self.parts_num)]
        self.__get_tags_fr_binary_tree__(root_backward, tags_backward)

        # Cut or pad.
        for i in range(self.parts_num):
            if self.max_seq_len > tokens_len:   # Pad.
                tags_forward[i] += [self.tag2id[i]["NULL"] for _ in range(self.max_seq_len - tokens_len)]
                tags_backward[i] += [self.tag2id[i]["NULL"] for _ in range(self.max_seq_len - tokens_len)]
            else:                               # Cut.
                tags_forward[i] = tags_forward[i][:self.max_seq_len]
                tags_backward[i] = tags_backward[i][:self.max_seq_len]

        # Tensor.
        for i in range(self.parts_num):
            tags_forward[i] = torch.LongTensor(tags_forward[i])
            tags_backward[i] = torch.LongTensor(tags_backward[i])

        return tags_forward + tags_backward

    def __get_spots_list__(self, sample):
        '''
            Get the spots list.
            Args:
                sample:                     {
                                                "tokens": ["a", "b", "c", ...],
                                                "relations": [
                                                    {
                                                        "label": "",
                                                        "label_array": ["", "", "", ...],
                                                        "subject": ["a"],
                                                        "object": ["c"],
                                                        "sub_span": (0, 1),
                                                        "obj_span": (2, 3)
                                                    }, ...
                                                ]
                                            }
            Returns:
                ent_spots:                  [(span_pos1, span_pos2)]
                rel_spots:                  [[(sub_span_pos1, sub_span_pos2), (obj_span_pos1, obj_span_pos1), isvalid=True]]
        '''
        ent_spots, rel_spots = set(), []

        for rel in sample["relations"]:
            sub_tok_span, obj_tok_span = rel["sub_span"], rel["obj_span"]
            if sub_tok_span[1] > self.max_seq_len or obj_tok_span[1] > self.max_seq_len: continue
            ent_spots.add(sub_tok_span)
            ent_spots.add(obj_tok_span)
            rel_spots.append([sub_tok_span, obj_tok_span, True])
        
        return list(ent_spots), list(rel_spots)

    def __sort_spots__(self, entity_spots_list, reverse=False):
        """ 
            Sort the entity_spots_list based on the first element.
            Args:
                entity_spots_list:      The entity spots list.
                reverse:                False -> From small to large. True -> From large to small.
        """
        def take_first(element):
            return element[0]
        
        entity_spots_list.sort(key=take_first, reverse=reverse)
    
    def __find_rel_between_tree_and_node__(self, root, node, rel_spots):
        """ 
            Find if there is a relation between the node in the tree and the new node
            Args:
                root:           The Binary Tree root
                node:           The new node
                rel_spots:      The relation spots list
            Return:
                parent:         The parent node of the new node if relation exists
                child_relation: The label of parent node if relation exists
        """
        parent, child_relation = None, ""
        if root:
            for rel in rel_spots:
                if rel[2] == False: continue
                if root.location == rel[0] and node.location == rel[1]:
                    child_relation = "1"        # The node is obj.
                    parent = root
                elif root.location == rel[1] and node.location == rel[0]:
                    child_relation = "2"        # The node is sub.
                    parent = root
            if parent == None and child_relation == "":
                parent, child_relation = self.__find_rel_between_tree_and_node__(root.firstChild, node, rel_spots)
            if parent == None and child_relation == "":
                parent, child_relation = self.__find_rel_between_tree_and_node__(root.nextSibling, node, rel_spots)
        return parent, child_relation
    
    def __build_binary_tree_fr_spots__(self, ent_spots, rel_spots):
        """ 
            Build binary tree from sorted entity spots and rel spots.
            Args:
                ent_spots:                  [(span_pos1, span_pos2)]
                rel_spots:                  [[(sub_span_pos1, sub_span_pos2), (obj_span_pos1, obj_span_pos1), isValid]]
            Returns:
                root:                       The root node of the relation binary tree.
        """
        root = None

        while len(ent_spots) != 0:
            # Get the ahead entity in the sentence, create a tree node from it.
            # Since the ent_spots has been sorted, we simply get the first one.
            node = BinaryTree(ent_spots[0])
            ent_spots.remove(ent_spots[0])

            # Find all children of the node.
            # Strictly promise the location sort of sibling nodes.
            remove_ents, relation_flag = [], False
            for i in range(len(ent_spots)):
                child_pair = ent_spots[i]
                for rel in rel_spots:
                    if rel[2] == False: continue        # If valid.
                    if node.location == rel[0] and child_pair == rel[1]:
                        child_node = BinaryTree(child_pair)
                        child_relation = "1"            # Means that child is obj.
                        node.insertChild(child_node, child_relation)
                        rel[2], relation_flag = False, True
                        remove_ents.append(child_pair)  # The entity to be removed.
                        break                           # Avoid two relations which have the same sub and the same obj
                    elif node.location == rel[1] and child_pair == rel[0]:
                        child_node = BinaryTree(child_pair)
                        child_relation = "2"            # Means that child is sub.
                        node.insertChild(child_node, child_relation)
                        rel[2], relation_flag = False, True
                        remove_ents.append(child_pair)  # The entity to be removed.
                        break
            
            # Insert the node into Binary tree
            if not root:
                root = node
            else:
                parent_node, child_relation = self.__find_rel_between_tree_and_node__(root, node, rel_spots)
                if parent_node != None and child_relation != "":
                    parent_node.insertChild(node, child_relation)
                else:
                    if relation_flag:
                        root.insertSibling(node, "brother")

            # Remove the entities added to the binary tree.
            for item in remove_ents:
                ent_spots.remove(item)
    
        return root
        
    def __get_tags_fr_binary_tree__(self, root, tags):
        """ 
            Tag the sentence according to the relation tree.
            Args:
                root:           The Binary Tree root.
                tags:           The tag list. [4, seq_len]
            Returns:
        """
        if root != None:
            pair = root.location
            changed = False
            for i in range(pair[0], pair[1]):
                if tags[0][i] != self.tag2id[0]["O"]: changed = True
            if changed == False:
                childRelation, siblingRelation = "None", "None"

                if root.childRelation != "": 
                    childRelation = root.childRelation
                if root.siblingRelation != "":
                    siblingRelation = root.siblingRelation

                # BIES+child-lable+parent-left-label+parent-right-label
                if pair[1] - pair[0] == 1:
                    tags[0][pair[0]] = self.tag2id[0]["S"]
                else:
                    tags[0][pair[0]] = self.tag2id[0]["B"]
                    tags[0][pair[1]-1] = self.tag2id[0]["E"]
                    tags[0][pair[0]+1:pair[1]-1] = [self.tag2id[0]["I"] for _ in range(pair[1] - pair[0] - 2)]
                tags[1][pair[0]:pair[1]] = [self.tag2id[1][root.parent] for _ in range(pair[1] - pair[0])]
                tags[2][pair[0]:pair[1]] = [self.tag2id[2][childRelation] for _ in range(pair[1] - pair[0])]
                tags[3][pair[0]:pair[1]] = [self.tag2id[3][siblingRelation] for _ in range(pair[1] - pair[0])]
            
            self.__get_tags_fr_binary_tree__(root.firstChild, tags)
            self.__get_tags_fr_binary_tree__(root.nextSibling, tags)
    
    def decode_rel_fr_bitt_tag(self, tokens, tags):
        """ 
            BiTT tags decoder.
            Args:
                tokens:         ["a", "b", "c", ...]
                tags:           (8, tags_len)
            Returns:
                rel_list:       [{
                                    "subject": subject tokens,
                                    "object": object tokens,
                                    "sub_span": subject location pair,
                                    "obj_span": object location pair
                                }, ...]
        """
        # Build the relation tree.
        root_forward = self.__build_relation_tree__(tokens, tags[:self.parts_num], forward=True)
        root_backward = self.__build_relation_tree__(tokens, tags[self.parts_num:], forward=False)

        # Get pred relation set.
        relations_forward, relations_backward = set(), set()
        self.__get_relation_fr_tree__(root_forward, relations_forward)
        self.__get_relation_fr_tree__(root_backward, relations_backward)
        relations = relations_forward.union(relations_backward)

        # Get rel_list.
        rel_list = []
        for relation in list(relations):
            spans = relation.split("-")
            rel_list.append({
                "subject": tokens[int(spans[0]):int(spans[1])], "object": tokens[int(spans[2]):int(spans[3])],
                "sub_span": (int(spans[0]), int(spans[1])), "obj_span": (int(spans[2]), int(spans[3]))
            })
        
        return rel_list
        
        
    def __build_relation_tree__(self, tokens, tags, forward):
        """ 
            Build the relation tree according to the tag and tokens. (Forward.)
            Args:
                text:           ["a", "b", "c", ...]
                tags:           (4, tags_len)
                forward:        True -> Forward, False -> Backward.
            Return:
                root:           the Binary tree
        """
        tokens_len, tags_len = len(tokens), tags[0].size(0)
        seq_len = min(tokens_len, tags_len)
        visit_flag = [False for _ in range(seq_len)]

        # Find all root in the forest.
        root_indexes = []
        if forward: # From right to left.
            cursor = seq_len - 1
            while cursor >= 0:
                if tags[1][cursor] in self.root_tags and tags[0][cursor] in self.begin_tags:
                    for i in range(cursor + 1, seq_len):
                        if tags[0][i] in self.inner_tags: continue
                        root_indexes.append((cursor, i))
                        break
                cursor -= 1
        else:       # From left to right.
            cursor = 0
            while cursor < seq_len:
                if tags[1][cursor] in self.root_tags and tags[0][cursor] in self.begin_tags:
                    for i in range(cursor + 1, seq_len):
                        if tags[0][i] in self.inner_tags: continue
                        root_indexes.append((cursor, i))
                        break
                cursor += 1
        
        # Build the tree.
        # Take the situation that two same entities are in a sentence into account.
        root, brother = None, None
        for root_index in root_indexes:
            root = BinaryTree(root_index)
            self.recurrent = 0
            self.__build_child_tree__(root, tags, visit_flag, seq_len, forward, True)
            if brother:
                root.insertSibling(brother, "brother")
            brother = root
        
        return root
    
    def __build_child_tree__(self, root, tags, visit_flag, seq_len, forward, isRoot):
        """ 
            Build a child tree in the forest.
            Args:
                root:           The root of the child tree.
                tags:           (4, tags_len)
                visit_flag:     (tags_len)
                seq_len:        The seq length.
                forward:        True -> Forward, False -> Backward.
                isRoot:         If this node a root node in the forest.
            Return:
        """
        if not visit_flag[root.location[0]]:
            visit_flag[root.location[0]] = True
        else:
            print("Waring: there is something wrong in building child tree!")
            return
        
        self.recurrent += 1
        if self.recurrent >= 1000:
            print("Warning: Died recurrent!")
            return
        
        # Left Tree, if there exists child node.
        if tags[2][root.location[0]] not in (self.none_tags + self.other_tags):
            parent_label = self.id2tag[2][tags[2][root.location[0]]].split("-")
            assert len(parent_label) == 1
            isFound = False
            if forward:     # Find the node on the right.
                for i in range(root.location[1], seq_len):
                    # Find the begin label and nonvisited label.
                    if tags[0][i] in self.begin_tags and visit_flag[i] == False:
                        # Find the label linked to the parent label.
                        child_label = self.id2tag[1][tags[1][i]].split("-")
                        if len(child_label) != 2: continue
                        if child_label[0] == "left":
                            if abs(int(child_label[1]) - int(parent_label[0])) == 1:
                                # We find the child node.
                                isFound = True
                                for j in range(i + 1, seq_len + 1):
                                    if j != seq_len:
                                        if tags[0][j] in self.inner_tags:
                                            continue
                                    child_node = BinaryTree((i, j))
                                    root.insertChild(child_node, self.id2tag[2][tags[2][root.location[0]]])
                                    assert root.firstChild.parent == self.id2tag[1][tags[1][i]]
                                    break
                                if root.firstChild == None:
                                    print("Warning: Cannot find the child node while finding right child node!")
                                    break
                                self.__build_child_tree__(root.firstChild, tags, visit_flag, seq_len, forward, False)
                                break
                # If we didn't find child node and current node is not root, we find child node on the left
                if isFound == False and isRoot == False:
                    #print("left2")
                    cursor = root.location[0] - 1
                    #print("cursor: ", cursor)
                    while cursor >= 0:
                        # Find the begin label and nonvisited label
                        if tags[0][cursor] in self.begin_tags and visit_flag[cursor] == False:
                            # Find the label linked to the parent label
                            child_label = self.id2tag[1][tags[1][cursor]].split("-")
                            if len(child_label) != 2: 
                                cursor -= 1
                                continue
                            if child_label[0] == "left":
                                if abs(int(child_label[1]) - int(parent_label[0])) == 1:
                                    # We find the child node
                                    #print("We find the child node")
                                    isFound = True
                                    for j in range(cursor + 1, root.location[0] + 1):
                                        if j!= root.location[0]:
                                            if tags[0][j] in self.inner_tags: continue
                                        child_node = BinaryTree((cursor, j))
                                        root.insertChild(child_node, self.id2tag[2][tags[2][root.location[0]]])
                                        assert root.firstChild.parent == self.id2tag[1][tags[1][cursor]]
                                        break
                                    if root.firstChild == None:
                                        print("Warning: Cannot find the child node while finding left child node!")
                                        break
                                    #print("Start 2!", self.recurrent)
                                    self.__build_child_tree__(root.firstChild, tags, visit_flag, seq_len, forward, False)
                                    #print("End 2!", self.recurrent)
                                    break
                        cursor -= 1
            else:           # Find the node on the left.
                cursor = root.location[0] - 1
                #print("cursor: ", cursor)
                while cursor >= 0:
                    # Find the begin label and nonvisited label
                    if tags[0][cursor] in self.begin_tags and visit_flag[cursor] == False:
                        # Find the label linked to the parent label
                        child_label = self.id2tag[1][tags[1][cursor]].split("-")
                        if len(child_label) != 2: 
                            cursor -= 1
                            continue
                        if child_label[0] == "left":
                            if abs(int(child_label[1]) - int(parent_label[0])) == 1:
                                # We find the child node
                                #print("We find the child node")
                                isFound = True
                                for j in range(cursor + 1, root.location[0] + 1):
                                    if j!= root.location[0]:
                                        if tags[0][j] in self.inner_tags: continue
                                    child_node = BinaryTree((cursor, j))
                                    root.insertChild(child_node, self.id2tag[2][tags[2][root.location[0]]])
                                    assert root.firstChild.parent == self.id2tag[1][tags[1][cursor]]
                                    break
                                if root.firstChild == None:
                                    print("Warning: Cannot find the child node while finding left child node!")
                                    break
                                #print("Start 2!", self.recurrent)
                                self.__build_child_tree__(root.firstChild, tags, visit_flag, seq_len, forward, False)
                                #print("End 2!", self.recurrent)
                                break
                    cursor -= 1
                # If we didn't find child node and current node is not root, we find child node on the right
                if isFound == False and isRoot == False:
                    #print("left2")
                    for i in range(root.location[1], seq_len):
                        # Find the begin label and nonvisited label
                        if tags[0][i] in self.begin_tags and visit_flag[i] == False:
                            # Find the label linked to the parent label
                            child_label = self.id2tag[1][tags[1][i]].split("-")
                            if len(child_label) != 2: continue
                            if child_label[0] == "left":
                                if abs(int(child_label[1]) - int(parent_label[0])) == 1:
                                    # We find the child node
                                    isFound = True
                                    for j in range(i + 1, seq_len + 1):
                                        if j != seq_len:
                                            if tags[0][j] in self.inner_tags: continue
                                        child_node = BinaryTree((i, j))
                                        root.insertChild(child_node, self.id2tag[2][tags[2][root.location[0]]])
                                        assert root.firstChild.parent == self.id2tag[1][tags[1][i]]
                                        break
                                    if root.firstChild == None:
                                        print("Warning: Cannot find the child node while finding right child node!")
                                        break
                                    self.__build_child_tree__(root.firstChild, tags, visit_flag, seq_len, forward, False)
                                    break
        
        # Right Tree, if there exists sibling node.
        if tags[3][root.location[0]] not in (self.none_tags + self.other_tags) and tags[3][root.location[0]] != 5:
            parent_label = self.id2tag[3][tags[3][root.location[0]]].split("-")
            #print("parent_label: ", parent_label)
            assert len(parent_label) == 1
            isFound = False
            if forward:     # Find the node on the right.
                for i in range(root.location[1], seq_len):
                    # Find the begin label and nonvisited label
                    if tags[0][i] in self.begin_tags and visit_flag[i] == False:
                        # Find the label linked to the parent label
                        sibling_label = self.id2tag[1][tags[1][i]].split("-")
                        if len(sibling_label) != 2: continue
                        if sibling_label[0] == "right" and sibling_label[1] != "brother":
                            #print("sibling_label: ", sibling_label[1])
                            #print("parent_label: ", parent_label[0])
                            if abs(int(sibling_label[1]) - int(parent_label[0])) == 1:
                                # We find the child node
                                isFound = True
                                for j in range(i + 1, seq_len + 1):
                                    if j != seq_len:
                                        if tags[0][j] in self.inner_tags: continue
                                    sibling_node = BinaryTree((i, j))
                                    root.insertSibling(sibling_node, self.id2tag[3][tags[3][root.location[0]]])
                                    assert root.nextSibling.parent == self.id2tag[1][tags[1][i]]
                                    break
                                if root.nextSibling == None:
                                    print("Warning: Cannot find the sibling node while finding right sibling node!")
                                    break
                                self.__build_child_tree__(root.nextSibling, tags, visit_flag, seq_len, forward, False)
                                break
            else:           # Find the node on the left.
                cursor = root.location[0] - 1
                while cursor >= 0:
                    # Find the begin label and nonvisited label
                    if tags[0][cursor] in self.begin_tags and visit_flag[cursor] == False:
                        # Find the label linked to the parent label
                        sibling_label = self.id2tag[1][tags[1][cursor]].split("-")
                        if len(sibling_label) != 2: 
                            cursor -= 1
                            continue
                        if sibling_label[0] == "right" and sibling_label[1] != "brother":
                            if abs(int(sibling_label[1]) - int(parent_label[0])) == 1:
                                # We find the child node
                                isFound = True
                                for j in range(cursor + 1, root.location[0] + 1):
                                    if j != root.location[0]:
                                        if tags[0][j] in self.inner_tags: continue
                                    sibling_node = BinaryTree((cursor, j))
                                    root.insertSibling(sibling_node, self.id2tag[3][tags[3][root.location[0]]])
                                    assert root.nextSibling.parent == self.id2tag[1][tags[1][cursor]]
                                    break
                                if root.nextSibling == None:
                                    print("Warning: Cannot find the sibling node while finding right sibling node!")
                                    break
                                self.__build_child_tree__(root.nextSibling, tags, visit_flag, seq_len, forward, False)
                                break
                    cursor -= 1
        
        return
    
    def __get_relation_fr_tree__(self, root, relation_set):
        """ 
            Get all predicted relations according the tree we had built.
            Args:
                root:           The root of the Binary tree
                relation_set:   All relations (sub_span_1-sub_span_2-obj_span_1-obj_span2)
            Returns:
        """
        if not root: return
        child_list = root.getChildList()
        if len(child_list):
            for child in child_list:
                label_list = child["label"].split("-")
                assert len(label_list) == 1
                if label_list[0] == "1":
                    relation_set.add("{}-{}-{}-{}".format(root.location[0], root.location[1], child["location"][0], child["location"][1]))
                else:
                    relation_set.add("{}-{}-{}-{}".format(child["location"][0], child["location"][1], root.location[0], root.location[1]))
        self.__get_relation_fr_tree__(root.firstChild, relation_set)
        self.__get_relation_fr_tree__(root.nextSibling, relation_set)
        return

class BiTTDataMaker():
    def __init__(self, tokenizer, tagger: BidirectionalTreeTaggingScheme, tag2id, label2array):
        """ DataMaker for Bidirectional Tree Tagging model. """
        self.tokenizer = tokenizer
        self.tagger = tagger
        self.rel2id = tag2id
        self.rel2array = label2array
    
    def get_indexed_data(self, data, mode, seq_max_len, label_max_len):
        """ 
            Get the indexed data.
            Args:
                data:               {
                                        "tokens": ["a", "b", "c", ...],
                                        "relations": [
                                            {
                                                "label": "",
                                                "label_array": ["", "", "", ...],
                                                "subject": ["a"],
                                                "object": ["c"],
                                                "sub_span": (0, 1),
                                                "obj_span": (2, 3)
                                            }, ...
                                        ]
                                    }
                mode:               The mode of data. 0 -> train, 1 -> valid.
                seq_max_len:        Max length of the src_ids.
                label_max_len:      Max length of label array.
            Returns:
                [(src_ids, seg_ids, mask_ids, tags, rel_id, sample)]
        """
        #print("Get indexed data...")
        indexed_data = []
        tokens, relations = data["tokens"], data["relations"]

        for rel, rel_id in self.rel2id.items():
            rel_tokens = self.rel2array[rel]
            rel_tokens_len = len(rel_tokens)

            if rel_tokens_len > label_max_len:
                rel_tokens = rel_tokens[0:label_max_len]
                rel_mask_index = label_max_len + 1
            else:
                rel_tokens = rel_tokens + ["[PAD]"] * (label_max_len - rel_tokens_len)
                rel_mask_index = rel_tokens_len + 1
        
            src_res = self.tokenizer(rel_tokens, tokens, is_split_into_words=True, truncation="only_second", max_length=seq_max_len, padding="max_length")
            src_ids, seg_ids, mask_ids = src_res["input_ids"], src_res["token_type_ids"], src_res["attention_mask"]
            mask_ids[rel_mask_index:label_max_len+1] = [0 for i in range(label_max_len - rel_mask_index + 1)]

            # Tagging.
            sample = {
                "tokens": tokens,
                "relations": [relation for relation in relations if relation["label"] == rel and relation["sub_span"][1] <= seq_max_len and relation["obj_span"][1] <= seq_max_len]
            }
            tags = self.tagger.encode_rel_to_bitt_tag(sample)

            indexed_data.append((src_ids, seg_ids, mask_ids, tags, rel_id, sample))
        
        return indexed_data

class FewBiTT(nn.Module):
    def __init__(self, args):
        """ 
            Fewshot version for Bidirectional Tree Tagging model.

            Args:
                encoder:            A Bert Encoder.
                map_hidden_size:    The hidden size of mapping layer.
                dist_type:          The type of calculating disance. ["dot", "euclidean"]
                label_max_length:   The max length of label inference.
                tagger:             BiTT Tagging Scheme
        """
        super(FewBiTT, self).__init__()
        self.encoder = args.encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.map_hidden_size = args.map_hidden_size
        self.dist_type = args.dist_type
        self.label_max_length = args.label_max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parts_num = args.tagger.parts_num
        self.tags_num = args.tagger.tags_num
        self.tags_weight = args.tagger.tags_weight
        self.parts_weight = args.tagger.parts_weight

        # Prototype.
        self.prototype = [
            torch.stack(
                [torch.zeros(self.map_hidden_size, device=self.device) for _ in range(self.tags_num[i % self.parts_num])], 0
        ) for i in range(self.parts_num * 2)]
        self.forget_rate = 0.9
        
        # Mapping Layer.
        self.mapping_fc = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.map_hidden_size) for _ in range(self.parts_num * 2)]
        )

        # Drop out.
        self.dropout = nn.Dropout()

        # Cost.
        self.cost = [nn.CrossEntropyLoss(weight=torch.Tensor(self.tags_weight[i])).to(self.device) for i in range(self.parts_num)]

    def forward(self, support, query):
        # Support.
        support_emb = self.encoder(support["src_ids"], support["mask_ids"], support["seg_ids"])[:, self.label_max_length + 2:, :]
        support_hidden = self.dropout(support_emb)
        support_hidden = torch.stack([self.mapping_fc[i](support_hidden) for i in range(self.parts_num * 2)])

        # Query.
        query_emb = self.encoder(query["src_ids"], query["mask_ids"], query["seg_ids"])[:, self.label_max_length + 2:, :]
        query_hidden = self.dropout(query_emb)
        query_hidden = torch.stack([self.mapping_fc[i](query_hidden) for i in range(self.parts_num * 2)])

        logits, pred = [[] for i in range(self.parts_num * 2)], [[] for i in range(self.parts_num * 2)]
        current_support_num, current_query_num = 0, 0

        for index, support_samples_num in enumerate(support["samples_num"]):
            query_samples_num = query["samples_num"][index]
            for i in range(self.parts_num * 2):
                logits[i].append(
                    self.__get_nearest_dist__(
                        support_hidden[i][current_support_num:current_support_num+support_samples_num],
                        support["tags"][i][current_support_num:current_support_num+support_samples_num],
                        query_hidden[i][current_query_num:current_query_num+query_samples_num],
                        part_index = i
                    )
                )
            current_query_num += query_samples_num
            current_support_num += support_samples_num
        
        for i in range(self.parts_num * 2):
            logits[i] = torch.cat(logits[i], 0)
            _, pred[i] = torch.max(logits[i], -1)
        
        return logits, pred
    
    def __dist__(self, x, y, dim):
        torch.cuda.empty_cache()
        if self.dist_type == "dot":
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)
    
    def __get_nearest_dist__(self, support_hidden, tag, query_hidden, part_index):
        """ 
            Get the nearest distance.
            Args:
                embedding:          (support_num, seq_len_support -> seq_S, hidden_size)
                tag:                (support_num, seq_S)
                query_hidden:       (query_num, seq_len_query -> seq_Q, hidden_size)
                part_index:         The index of tag parts.
            Returns:
                logits:             (query_num, seq_Q, tag_dim)
        """
        seq_len, hidden_size = support_hidden.size(1), support_hidden.size(2)
        support_num, query_num = support_hidden.size(0), query_hidden.size(0)

        # Get the distance matrix between support set and query set.
        S = support_hidden.view(-1, hidden_size).unsqueeze(0)   # (1, support_num * seq_S -> tokens_S, hidden_size)
        Q = query_hidden.view(-1, hidden_size).unsqueeze(1)     # (query_num * seq_Q -> tokens_Q, 1, hidden_size)
        dist = self.__dist__(S, Q, dim=-1)                      # (tokens_Q, tokens_S)

        # Get the top distance for every tag.
        tag_dim = self.tags_num[part_index % self.parts_num]
        nearest_dist, T = [[] for i in range(tag_dim)], tag.view(-1)
        for label in range(tag_dim):
            dist_label = dist[:, T==label]                                                          # (tokens_Q, *)
            if dist_label.size(1) != 0:
                nearest_dist[label], _ = torch.max(dist_label, dim=1)                               # (tokens_Q)
            else:
                prototype_label = self.prototype[part_index][label]                                 # (hidden_size)
                nearest_dist[label] = self.__dist__(prototype_label, Q.view(-1, hidden_size), -1)   # (tokens_Q)
        
        # Get logits.
        logits = torch.stack(nearest_dist, dim=1).view(query_num, seq_len, tag_dim)                 # (query_num, seq_len, tag_dim)

        # Updata prototype.
        for label in range(tag_dim):
            hidden_label = S.view(-1, hidden_size)[T==label, :] # (*, hidden_size)
            if hidden_label.size(0) != 0:
                hidden_avg = hidden_label.sum(0) / hidden_label.size(0)
                self.prototype[part_index][label] *= (1 - self.forget_rate)
                self.prototype[part_index][label] += (self.forget_rate * hidden_avg.detach())
        
        """ # Get the distance matrix between prototype and query set.
        tag_dim, T = self.tags_num[part_index % self.parts_num], tag.view(-1)
        dist_pro = self.__dist__(self.prototype[part_index], Q, dim=-1) # (tokens_Q, tag_dim)
        dist_pro = torch.transpose(dist_pro, 0, 1).view(tag_dim, -1, 1) # (tag_dim, tokens_Q, 1)
        
        # Get the top distance for every tag.
        tag_indexes = torch.linspace(0, tag_dim - 1, tag_dim).long().to(self.device)                # (tag_dim), [0, 1, 2]
        tag_indexes = tag_indexes.view(-1, 1).expand(-1, support_num * seq_len)                     # (tag_dim, tokens_S), [[0, 0], [1, 1], [2, 2]]
        tag_mask = (T == tag_indexes).long().contiguous().view(tag_dim, -1, support_num * seq_len)  # (tag_dim, 1, tokens_S)
        dist_mask = dist.view(1, -1, support_num * seq_len) * tag_mask + dist_pro * (1 - tag_mask)  # (tag_dim, tokens_Q, tokens_S)
        dist_max, _ = torch.max(dist_mask, -1)                                                      # (tag_dim, tokens_Q)

        # Get logits.
        logits = torch.transpose(dist_max, 0, 1).view(query_num, seq_len, tag_dim)  # (query_num, seq_Q, tag_dim)

        # Update prototype.
        hidden_mask = S * tag_mask.view(tag_dim, support_num * seq_len, 1)                      # (tag_dim, tokens_S, hidden_size)
        hidden_sum = hidden_mask.sum(1)                                                         # (tag_dim, hidden_size)
        hidden_len = tag_mask.view(tag_dim, support_num * seq_len).sum(1).view(-1, 1)           # (tag_dim, 1)
        hidden_avg = hidden_sum / (hidden_len + 1e-5)                                           # (tag_dim, hidden_size)
        update_mask = torch.max(tag_mask.view(tag_dim, -1), -1)[0].view(-1, 1)                  # (tag_dim, 1)
        self.prototype[part_index] *= (1 - self.forget_rate * update_mask)                      # (tag_dim, hidden_size)
        self.prototype[part_index] += (self.forget_rate * update_mask * hidden_avg.detach())    # (tag_dim, hidden_size) """

        return logits

    def loss(self, logits, label, quiet=True):
        '''
            Args:

                logits:     Logits with the size (8, query_num, seq_len, class_num)
                label:      Label with the size (8, query_num, seq_len).

            Returns: 

                [Loss] (A single value)
        '''
        loss = []
        for i in range(self.parts_num * 2):
            N = logits[i].size(-1)
            loss.append(self.cost[i % self.parts_num](logits[i].view(-1, N), label[i].view(-1)))
        
        if not quiet:
            str_format = "Loss of every part: "
            for i in range(self.parts_num * 2):
                str_format += "{}, ".format(loss[i])
            print(str_format[:-2])
        
        total_loss = 0.0
        for i in range(self.parts_num * 2):
            total_loss += (self.parts_weight[i % self.parts_num] * loss[i])
        return total_loss
    
class BiTTMetricsCalculator():
    def __init__(self, tagger: BidirectionalTreeTaggingScheme):
        self.tagger = tagger
        self.parts_num = tagger.parts_num
        self.accs_keys = [
            "F-part1", "F-part2", "F-part3", "F-part4",
            "B-part1", "B-part2", "B-part3", "B-part4",
        ]
    
    def get_accs(self, preds, labels):
        '''
            The accuracy of all pred labels of a sample are right.

            Args:

                preds:      Preds with the size (8, query_num, seq_len).
                label:      Label with the size (8, query_num, seq_len).

            Returns: 

                [accs] (A dict)
        '''
        accs = {}

        for index, key in enumerate(self.accs_keys):
            pred_ids, label_ids = preds[index], labels[index]

            correct_tag_num = torch.sum(torch.eq(label_ids, pred_ids).float(), dim=1)
            
            # correct_tag_num == seq_len.
            sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * label_ids.size()[-1]).float()
            sample_acc = torch.mean(sample_acc_)
            
            accs[self.accs_keys[index]] = sample_acc
        
        return accs
    
    def get_rel_pgc(self, samples, preds):
        '''
            Get pred, gold and correct.

            Args:

                samples:    Sample dict.
                            {
                                "tokens": ["a", "b", "c", ...],
                                "relations": [
                                    {
                                        "label": "",
                                        "label_array": ["", "", "", ...],
                                        "subject": ["a"],
                                        "object": ["c"],
                                        "sub_span": (0, 1),
                                        "obj_span": (2, 3)
                                    }, ...
                                ]
                            }
                preds:      Preds with the size (8, query_num, seq_len).

            Returns: 

                (pred, gold, correct)
        '''
        seq_len = preds[0].size(1)
        #print("seq_len: ", seq_len)
        correct_num, pred_num, gold_num = 0, 0, 0

        for index in range(len(samples)):
            sample = samples[index]
            tokens = sample["tokens"]
            tags = [preds[i][index] for i in range(self.parts_num * 2)]

            pred_rel_list = self.tagger.decode_rel_fr_bitt_tag(tokens, tags)

            gold_rel_list = sample["relations"]

            pred_rel_set = set([
                "{}, {}, {}, {}".format(
                    rel["sub_span"][0], rel["sub_span"][1], rel["obj_span"][0], rel["obj_span"][1]
                ) for rel in pred_rel_list
            ])
            gold_rel_set = set([
                "{}, {}, {}, {}".format(
                    rel["sub_span"][0], rel["sub_span"][1], rel["obj_span"][0], rel["obj_span"][1]
                ) for rel in gold_rel_list if rel["sub_span"][1] <= seq_len and rel["obj_span"][1] <= seq_len
            ])

            correct_num += len(pred_rel_set.intersection(gold_rel_set))
            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)
        
        return pred_num, gold_num, correct_num