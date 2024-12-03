import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from collections import deque
import csv

# Узел бинарного дерева поиска
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if not self.root:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    def _insert(self, current, key):
        if key < current.key:
            if current.left is None:
                current.left = Node(key)
            else:
                self._insert(current.left, key)
        else:
            if current.right is None:
                current.right = Node(key)
            else:
                self._insert(current.right, key)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, current, key):
        if current is None or current.key == key:
            return current
        if key < current.key:
            return self._search(current.left, key)
        return self._search(current.right, key)

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, current, key):
        if current is None:
            return current
        if key < current.key:
            current.left = self._delete(current.left, key)
        elif key > current.key:
            current.right = self._delete(current.right, key)
        else:
            if current.left is None:
                return current.right
            elif current.right is None:
                return current.left
            temp = self._min_value_node(current.right)
            current.key = temp.key
            current.right = self._delete(current.right, temp.key)
        return current

    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def height(self):
        return self._height(self.root)

    def _height(self, current):
        if current is None:
            return 0
        return 1 + max(self._height(current.left), self._height(current.right))

    def in_order(self):
        return self._in_order(self.root)

    def _in_order(self, current):
        if current is None:
            return []
        return self._in_order(current.left) + [current.key] + self._in_order(current.right)

    def pre_order(self):
        return self._pre_order(self.root)

    def _pre_order(self, current):
        if current is None:
            return []
        return [current.key] + self._pre_order(current.left) + self._pre_order(current.right)

    def post_order(self):
        return self._post_order(self.root)

    def _post_order(self, current):
        if current is None:
            return []
        return self._post_order(current.left) + self._post_order(current.right) + [current.key]

    def bfs(self):
        return self._bfs(self.root)

    def _bfs(self, current):
        if current is None:
            return []
        queue = deque([current])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node.key)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return result

# Узел AVL-дерева
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, current, key):
        if not current:
            return AVLNode(key)
        if key < current.key:
            current.left = self._insert(current.left, key)
        elif key > current.key:
            current.right = self._insert(current.right, key)
        else:
            return current

        current.height = 1 + max(self._get_height(current.left), self._get_height(current.right))
        balance = self._get_balance(current)

        if balance > 1 and key < current.left.key:
            return self._rotate_right(current)
        if balance < -1 and key > current.right.key:
            return self._rotate_left(current)
        if balance > 1 and key > current.left.key:
            current.left = self._rotate_left(current.left)
            return self._rotate_right(current)
        if balance < -1 and key < current.right.key:
            current.right = self._rotate_right(current.right)
            return self._rotate_left(current)

        return current

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _get_height(self, current):
        if not current:
            return 0
        return current.height

    def _get_balance(self, current):
        if not current:
            return 0
        return self._get_height(current.left) - self._get_height(current.right)

    def height(self):
        return self._get_height(self.root)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, current, key):
        if current is None or current.key == key:
            return current
        if key < current.key:
            return self._search(current.left, key)
        return self._search(current.right, key)

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, current, key):
        if not current:
            return current
        if key < current.key:
            current.left = self._delete(current.left, key)
        elif key > current.key:
            current.right = self._delete(current.right, key)
        else:
            if current.left is None:
                temp = current.right
                current = None
                return temp
            elif current.right is None:
                temp = current.left
                current = None
                return temp
            temp = self._min_value_node(current.right)
            current.key = temp.key
            current.right = self._delete(current.right, temp.key)
        if current is None:
            return current
        current.height = 1 + max(self._get_height(current.left), self._get_height(current.right))
        balance = self._get_balance(current)
        if balance > 1 and self._get_balance(current.left) >= 0:
            return self._rotate_right(current)
        if balance > 1 and self._get_balance(current.left) < 0:
            current.left = self._rotate_left(current.left)
            return self._rotate_right(current)
        if balance < -1 and self._get_balance(current.right) <= 0:
            return self._rotate_left(current)
        if balance < -1 and self._get_balance(current.right) > 0:
            current.right = self._rotate_right(current.right)
            return self._rotate_left(current)
        return current

    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def in_order(self):
        return self._in_order(self.root)

    def _in_order(self, current):
        if current is None:
            return []
        return self._in_order(current.left) + [current.key] + self._in_order(current.right)

    def pre_order(self):
        return self._pre_order(self.root)

    def _pre_order(self, current):
        if current is None:
            return []
        return [current.key] + self._pre_order(current.left) + self._pre_order(current.right)

    def post_order(self):
        return self._post_order(self.root)

    def _post_order(self, current):
        if current is None:
            return []
        return self._post_order(current.left) + self._post_order(current.right) + [current.key]

    def bfs(self):
        return self._bfs(self.root)

    def _bfs(self, current):
        if current is None:
            return []
        queue = deque([current])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node.key)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return result

# Узел красно-черного дерева
class RBNode:
    def __init__(self, key, color="RED"):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
        self.color = color

class RBTree:
    def __init__(self):
        self.NIL = RBNode(key=None, color="BLACK")
        self.root = self.NIL

    def insert(self, key):
        new_node = RBNode(key)
        new_node.left = self.NIL
        new_node.right = self.NIL
        self._insert(new_node)

    def _insert(self, z):
        y = None
        x = self.root
        while x != self.NIL:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right

        z.parent = y
        if y is None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

        z.color = "RED"
        self._fix_insert(z)

    def _fix_insert(self, z):
        while z != self.root and z.parent.color == "RED":
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == "RED":
                    z.parent.color = "BLACK"
                    y.color = "BLACK"
                    z.parent.parent.color = "RED"
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self._rotate_left(z)
                    z.parent.color = "BLACK"
                    z.parent.parent.color = "RED"
                    self._rotate_right(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == "RED":
                    z.parent.color = "BLACK"
                    y.color = "BLACK"
                    z.parent.parent.color = "RED"
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color = "BLACK"
                    z.parent.parent.color = "RED"
                    self._rotate_left(z.parent.parent)
        self.root.color = "BLACK"

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, current, key):
        if current == self.NIL or current.key == key:
            return current
        if key < current.key:
            return self._search(current.left, key)
        return self._search(current.right, key)

    def delete(self, key):
        self._delete(self.root, key)

    def _delete(self, current, key):
        z = self.NIL
        while current != self.NIL:
            if current.key == key:
                z = current
            if current.key <= key:
                current = current.right
            else:
                current = current.left
        if z == self.NIL:
            return
        y = z
        y_original_color = y.color
        if z.left == self.NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right == self.NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        if y_original_color == "BLACK":
            self._fix_delete(x)

    def _transplant(self, u, v):
        if u.parent == self.NIL:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, x):
        while x.left != self.NIL:
            x = x.left
        return x

    def _fix_delete(self, x):
        while x != self.root and x.color == "BLACK":
            if x == x.parent.left:
                w = x.parent.right
                if w.color == "RED":
                    w.color = "BLACK"
                    x.parent.color = "RED"
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color == "BLACK" and w.right.color == "BLACK":
                    w.color = "RED"
                    x = x.parent
                else:
                    if w.right.color == "BLACK":
                        w.left.color = "BLACK"
                        w.color = "RED"
                        self._rotate_right(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = "BLACK"
                    w.right.color = "BLACK"
                    self._rotate_left(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == "RED":
                    w.color = "BLACK"
                    x.parent.color = "RED"
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color == "BLACK" and w.left.color == "BLACK":
                    w.color = "RED"
                    x = x.parent
                else:
                    if w.left.color == "BLACK":
                        w.right.color = "BLACK"
                        w.color = "RED"
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = "BLACK"
                    w.left.color = "BLACK"
                    self._rotate_right(x.parent)
                    x = self.root
        x.color = "BLACK"

    def height(self):
        def _height(node):
            if node == self.NIL:
                return 0
            left_height = _height(node.left)
            right_height = _height(node.right)
            return max(left_height, right_height) + 1

        return _height(self.root)

    def in_order(self):
        return self._in_order(self.root)

    def _in_order(self, current):
        if current == self.NIL:
            return []
        return self._in_order(current.left) + [current.key] + self._in_order(current.right)

    def pre_order(self):
        return self._pre_order(self.root)

    def _pre_order(self, current):
        if current == self.NIL:
            return []
        return [current.key] + self._pre_order(current.left) + self._pre_order(current.right)

    def post_order(self):
        return self._post_order(self.root)

    def _post_order(self, current):
        if current == self.NIL:
            return []
        return self._post_order(current.left) + self._post_order(current.right) + [current.key]

    def bfs(self):
        return self._bfs(self.root)

    def _bfs(self, current):
        if current == self.NIL:
            return []
        queue = deque([current])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node.key)
            if node.left != self.NIL:
                queue.append(node.left)
            if node.right != self.NIL:
                queue.append(node.right)
        return result

# Функция для записи результатов обхода в файл
def write_traversal_to_file(filename, traversal):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(traversal)

# Построение графиков
def build_and_plot():
    bst = BST()
    avl = AVLTree()
    rb = RBTree()

    # Пример для BST с 10 ключами
    bstnum_keys = [5, 3, 7, 2, 4, 6, 8, 1, 9, 10]
    for key in bstnum_keys:
        bst.insert(key)
    print("BST In-order:", bst.in_order())
    print("BST Pre-order:", bst.pre_order())
    print("BST Post-order:", bst.post_order())
    print("BST BFS:", bst.bfs())

    # Пример для AVL с 10 ключами
    avlnum_keys = [5, 3, 7, 2, 4, 6, 8, 1, 9, 10]
    for key in avlnum_keys:
        avl.insert(key)
    print("AVL In-order:", avl.in_order())
    print("AVL Pre-order:", avl.pre_order())
    print("AVL Post-order:", avl.post_order())
    print("AVL BFS:", avl.bfs())

    # Пример для RB-Tree с 10 ключами
    rbnum_keys = [5, 3, 7, 2, 4, 6, 8, 1, 9, 10]
    for key in rbnum_keys:
        rb.insert(key)
    print("RB-Tree In-order:", rb.in_order())
    print("RB-Tree Pre-order:", rb.pre_order())
    print("RB-Tree Post-order:", rb.post_order())
    print("RB-Tree BFS:", rb.bfs())

    bst_keys = [random.randint(1, 100000) for _ in range(50000)]
    avl_keys = list(range(1, 50001))
    rb_keys = list(range(1, 50001))

    bst_heights = []
    avl_heights = []
    rb_heights = []
    x_vals = range(1000, 50001, 1000)

    for i, key in enumerate(bst_keys):
        bst.insert(key)
        if (i + 1) % 1000 == 0:
            bst_heights.append(bst.height())

    for i, key in enumerate(avl_keys):
        avl.insert(key)
        if (i + 1) % 1000 == 0:
            avl_heights.append(avl.height())

    for i, key in enumerate(rb_keys):
        rb.insert(key)
        if (i + 1) % 1000 == 0:
            rb_heights.append(rb.height())

    # Функция для логарифмической регрессии
    def log_func(x, a, b):
        return a * np.log(x) + b

    # Регрессия для BST
    bst_params, _ = curve_fit(log_func, x_vals, bst_heights)
    print(f"BST логарифмическая регрессия: y = {bst_params[0]:.4f} * ln(x) + {bst_params[1]:.4f}")

    # Регрессия для AVL
    avl_params, _ = curve_fit(log_func, x_vals, avl_heights)
    print(f"AVL логарифмическая регрессия: y = {avl_params[0]:.4f} * ln(x) + {avl_params[1]:.4f}")

    # Регрессия для RB
    rb_params, _ = curve_fit(log_func, x_vals, rb_heights)
    print(f"RB-Tree логарифмическая регрессия: y = {rb_params[0]:.4f} * ln(x) + {rb_params[1]:.4f}")

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # Общий график
    plt.plot(x_vals, bst_heights, label="BST", marker='o', linestyle='')
    plt.plot(x_vals, avl_heights, label="AVL", marker='x', linestyle='')
    plt.plot(x_vals, rb_heights, label="RB-Tree", marker='s', linestyle='')
    plt.plot(x_vals, log_func(np.array(x_vals), *bst_params), label="BST регрессия", linestyle='--')
    plt.plot(x_vals, log_func(np.array(x_vals), *avl_params), label="AVL регрессия", linestyle='--')
    plt.plot(x_vals, log_func(np.array(x_vals), *rb_params), label="RB-Tree регрессия", linestyle='--')
    plt.xlabel("Количество ключей")
    plt.ylabel("Высота дерева")
    plt.legend()
    plt.grid()
    plt.title("Анализ высоты деревьев")
    plt.savefig("tree_heights_combined.png")
    plt.close()

    # График для BST
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, bst_heights, label="BST", marker='o', linestyle='')
    plt.plot(x_vals, log_func(np.array(x_vals), *bst_params), label="BST регрессия", linestyle='--')
    plt.xlabel("Количество ключей")
    plt.ylabel("Высота дерева")
    plt.legend()
    plt.grid()
    plt.title("Анализ высоты BST")
    plt.savefig("bst_height.png")
    plt.close()

    # График для AVL
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, avl_heights, label="AVL", marker='x', linestyle='')
    plt.plot(x_vals, log_func(np.array(x_vals), *avl_params), label="AVL регрессия", linestyle='--')
    plt.xlabel("Количество ключей")
    plt.ylabel("Высота дерева")
    plt.legend()
    plt.grid()
    plt.title("Анализ высоты AVL")
    plt.savefig("avl_height.png")
    plt.close()

    # График для RB-Tree
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, rb_heights, label="RB-Tree", marker='s', linestyle='')
    plt.plot(x_vals, log_func(np.array(x_vals), *rb_params), label="RB-Tree регрессия", linestyle='--')
    plt.xlabel("Количество ключей")
    plt.ylabel("Высота дерева")
    plt.legend()
    plt.grid()
    plt.title("Анализ высоты RB-Tree")
    plt.savefig("rb_height.png")
    plt.close()

    # Запись обходов в файлы
    write_traversal_to_file('bst_in_order.csv', bst.in_order())
    write_traversal_to_file('bst_pre_order.csv', bst.pre_order())
    write_traversal_to_file('bst_post_order.csv', bst.post_order())
    write_traversal_to_file('bst_bfs.csv', bst.bfs())

    write_traversal_to_file('avl_in_order.csv', avl.in_order())
    write_traversal_to_file('avl_pre_order.csv', avl.pre_order())
    write_traversal_to_file('avl_post_order.csv', avl.post_order())
    write_traversal_to_file('avl_bfs.csv', avl.bfs())

    write_traversal_to_file('rb_in_order.csv', rb.in_order())
    write_traversal_to_file('rb_pre_order.csv', rb.pre_order())
    write_traversal_to_file('rb_post_order.csv', rb.post_order())
    write_traversal_to_file('rb_bfs.csv', rb.bfs())

if __name__ == "__main__":
    build_and_plot()
