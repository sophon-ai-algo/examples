/*==========================================================================
 * Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
============================================================================*/
//
// Created by yuan on 1/4/22.
//

#ifndef INFERENCE_FRAMEWORK_BMUTILITY_LIST_H
#define INFERENCE_FRAMEWORK_BMUTILITY_LIST_H

struct ListHead {
    struct ListHead *prev, *next;
};

#define INIT_LIST_HEAD(ptr) do {(ptr)->next = (ptr); (ptr)->prev = (ptr);} while (0)
#define list_for_each_next(pos, head)  for (pos = (head)->next; pos != (head); pos = pos->next)
#define list_for_each_prev(pos, head)  for (pos = (head)->prev; pos != (head); pos = pos->prev)
#define list_for_each_safe(pos, n, head) for (pos = (head)->next, n = pos->next; pos != (head); pos = n, n = pos->next)

#define LIST_HOST_ENTRY(address, type, field) ((type *)((char*)(address) - (size_t)(&((type *)0)->field)))
#define list_for_each_entry_next(pos, head, T, member)              \
    for (pos = LIST_HOST_ENTRY((head)->next, T, member); &pos->member != (head);  \
         pos = LIST_HOST_ENTRY(pos->member.next, T, member))



static __inline void __list_add(struct ListHead *newNode, struct ListHead *prev, struct ListHead *next) {
    next->prev = newNode;
    newNode->next = next;
    newNode->prev = prev;
    prev->next = newNode;
}

static __inline void __list_del(struct ListHead *prev, struct ListHead *next)
{
    next->prev = prev;
    prev->next = next;
}

static __inline void list_push_back(struct ListHead *newNode, struct ListHead *head) {
    INIT_LIST_HEAD(newNode);
    __list_add(newNode, head, head->next);
}

static __inline void list_push_front(struct ListHead *newNode, struct ListHead *head) {
    INIT_LIST_HEAD(newNode);
    __list_add(newNode, head->prev, head);
}

static __inline void list_del(struct ListHead *entry) {
    __list_del(entry->prev, entry->next);
}

static __inline int  list_empty(struct ListHead *head)
{
    return head->next == head;
}

static __inline struct ListHead* list_next(struct ListHead *head) {
    return head->next;
}



static __inline struct ListHead* list_prev(struct ListHead *head) {
    return head->prev;
}

#define list_front(head) list_prev(head)
#define list_back(head) list_next(head)


#endif //INFERENCE_FRAMEWORK_BMUTILITY_LIST_H
