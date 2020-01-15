#pragma once
#pragma once
#include<iostream>
#include<vector>
using namespace std;


#ifndef GG_H
#define GG_H

struct TreeNode {
	int val;
	TreeNode* left, *right;
	TreeNode() {}
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};


struct NodeNeighbor {
	int val;
	vector<NodeNeighbor*> neighbors;
	NodeNeighbor() {}
	NodeNeighbor(int _val) : val(_val) {}
	NodeNeighbor(int _val, vector<NodeNeighbor*> _neighbors) {
		val = _val;
		neighbors = _neighbors;
	}
};

/* structure of basic ll */
struct ListNode {
	int val;
	ListNode *next;
	ListNode() {}
	ListNode(int x) : val(x), next(NULL) {}
	ListNode(int x, ListNode* _next) : val(x), next(_next) {}
};

/* structure of ll which is circular: prev and next pointers for each node */
struct CircularNode {
	int val;
	CircularNode* left;
	CircularNode* right;

	CircularNode() {}

	CircularNode(int _val, CircularNode* _left, CircularNode* _right) {
		val = _val;
		left = _left;
		right = _right;
	}
};

/* Structure of ll node with random pointer */
struct RandomNode {
	int val;
	RandomNode* next;
	RandomNode* random;

	RandomNode() {}

	RandomNode(int _val, RandomNode* _next, RandomNode* _random) {
		val = _val;
		next = _next;
		random = _random;
	}
};

/* doubly linked list which in addition to the next and previous pointers */
class ChildNode {
public:
	int val;
	ChildNode* prev;
	ChildNode* next;
	ChildNode* child;

	ChildNode() {}

	ChildNode(int _val, ChildNode* _prev, ChildNode* _next, ChildNode* _child) {
		val = _val;
		prev = _prev;
		next = _next;
		child = _child;
	}
};

/* 707. Design Linked List */
class MyLinkedList {
	struct node {
		int value;
		node* next;
		node(int val) :value(val), next(nullptr) {}
	};
	int size;
	node* head;

public:

	/** Initialize your data structure here. */
	MyLinkedList() :size(0) {
		head = new node(0);
	}

	/** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
	int get(int index) {
		if (index < 0 || index >= size) return -1;
		node *curr = head->next;
		for (int i = 0; i<index; ++i) curr = curr->next;
		return curr->value;
	}

	/** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
	void addAtHead(int val) {
		node* curr = head->next;
		head->next = new node(val);
		head->next->next = curr;
		++size;
	}

	/** Append a node of value val to the last element of the linked list. */
	void addAtTail(int val) {
		node* curr = head;
		while (curr->next) curr = curr->next;
		curr->next = new node(val);
		++size;
	}

	/** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
	void addAtIndex(int index, int val) {
		if (index > size) return;
		node* curr = head;
		for (int i = 0; i < index; ++i) curr = curr->next;
		node* temp = curr->next;
		curr->next = new node(val);
		curr->next->next = temp;
		++size;
	}

	/** Delete the index-th node in the linked list, if the index is valid. */
	void deleteAtIndex(int index) {
		if (index < 0 || index >= size) return;
		node* curr = head;
		for (int i = 0; i < index; ++i) curr = curr->next;
		node* temp = curr->next;
		curr->next = temp->next;
		--size;
		delete temp;
	}


};

class Employee {
public:
	// It's the unique ID of each node.
	// unique id of this employee
	int id;
	// the importance value of this employee
	int importance;
	// the id of direct subordinates
	vector<int> subordinates;
};

struct ListNodeMultiLevel {
	int val;
	ListNodeMultiLevel* next, *prev, *child;
	ListNodeMultiLevel() {}
	ListNodeMultiLevel(int _val, ListNodeMultiLevel* _next,
		ListNodeMultiLevel *_prev, ListNodeMultiLevel *_child) {
		val = _val;
		next = _next;
		prev = _prev;
		child = _child;
	}
};

struct NaryTreeNode {
	int val;
	vector<NaryTreeNode*> children;
	NaryTreeNode() {}
	NaryTreeNode(int _val, vector<NaryTreeNode*> _children) {
		val = _val;
		children = _children;
	}
};

struct TreeNodeNext {
	int val;
	TreeNodeNext* left, *right, *next;
	TreeNodeNext() {}
	TreeNodeNext(int _val, TreeNodeNext* _left, TreeNodeNext* _right, TreeNodeNext* _next) {
		val = _val;
		left = _left;
		right = _right;
		next = _next;
	}
};

struct TrieNode {
	bool isWord;
	TrieNode* child[26];
	TrieNode() :isWord(false) {
		for (auto &a : child) a = NULL;
	}
};

/* structure of ll which is circular: prev and next pointers for each node */
struct CircularNode {
	int val;
	CircularNode* left;
	CircularNode* right;

	CircularNode() {}

	CircularNode(int _val, CircularNode* _left, CircularNode* _right) {
		val = _val;
		left = _left;
		right = _right;
	}
};

/* Structure of ll node with random pointer */
struct RandomNode {
	int val;
	RandomNode* next;
	RandomNode* random;

	RandomNode() {}

	RandomNode(int _val, RandomNode* _next, RandomNode* _random) {
		val = _val;
		next = _next;
		random = _random;
	}
};

class Interval {
public:
	int start;
	int end;

	Interval() {}

	Interval(int _start, int _end) {
		start = _start;
		end = _end;
	}
};

// ===========================================================

// ================ 12. DESIGN PROBLEMS ======================
/* 211. Add and Search Word - Data structure design -- TRIE */
/* Design a data structure that supports the following two operations:
* void addWord(word), bool search(word), search(word) can search a literal
* word or a regular expression string containing only letters a-z or .. A .
* means it can represent any one letter. */
class WordDictionary {
public:
	WordDictionary() {
		root = new TrieNode();
	}

	void addWord(string word) {
		TrieNode* p = root;
		for (auto c : word) {
			int i = c - 'a';
			if (!p->child[i]) p->child[i] = new TrieNode();
			p = p->child[i];
		}
		p->isWord = true;
	}

	bool searchHelper(string s, TrieNode* p, int ix) {
		if (ix == s.size()) return p->isWord;
		if (s[ix] == '.') {
			for (auto a : p->child) {
				if (a && searchHelper(s, a, ix + 1)) return true; 
			}
			return false; 
		}
		else {
			int i = s[ix] - 'a';
			if (p->child[i] && searchHelper(s, p->child[i], ix + 1)) return true; 
		}
		return false; 
	}

	bool search(string word) {
		return searchHelper(word, root, 0);
	}

private:
	TrieNode* root; 
};

/* 173. Binary Search Tree Iterator -- STACK */
/* Implement an iterator over a binary search tree (BST). Your iterator will be
* initialized with the root node of a BST. Calling next() will return the next 
* smallest number in the BST. */
class BSTIterator {
public:
	BSTIterator(TreeNode* root) {
		TreeNode* t = root;
		while (t) {
			st.push(t);
			t = t->left;
		}
	}

	/** @return the next smallest number */
	int next() {
		int res = 0;
		auto t = st.top(); st.pop();
		res = t->val;

		if (t->right) {
			t = t->right;
			while (t) {
				st.push(t);
				t = t->left;
			}
		}

		return res;
	}

	/** @return whether we have a next smallest number */
	bool hasNext() {
		return !st.empty();
	}

private:
	stack<TreeNode*> st;
};

/* 919. Complete Binary Tree Inserter */
/* A complete binary tree is a binary tree in which every level, except possibly the last,
* is completely filled, and all nodes are as far left as possible. Write a data structure 
* CBTInserter that is initialized with a complete binary tree and supports the following operations:
* (1) CBTInserter(TreeNode root) initializes the data structure on a given tree with head node root;
* (2) CBTInserter.insert(int v) will insert a TreeNode into the tree with value node.val = v 
*     so that the tree remains complete, and returns the value of the parent of the inserted TreeNode;
* (3) CBTInserter.get_root() will return the head node of the tree.  */
class CBTInserter {
public:
	CBTInserter(TreeNode* root) {
		tree.push_back(root);
		for (int i = 0; i < tree.size(); ++i) {
			if (tree[i]->left) tree.push_back(tree[i]->left);
			if (tree[i]->right) tree.push_back(tree[i]->right);
		}
	}

	int insert(int v) {
		int n = tree.size(); 
		TreeNode* newnode = new TreeNode(v); 
		tree.push_back(newnode); 
		if (n % 2) tree[(n - 1) / 2]->left = newnode; 
		else tree[(n - 1) / 2]->right = newnode; 
		return tree[(n - 1) / 2] ->val;
	}

	TreeNode* get_root() {
		return tree[0]; 
	}

private: 
	vector<TreeNode*> tree; 
};

/* 622. Design Circular Queue */
/* Design your implementation of the circular queue. The circular queue is a linear 
* data structure in which the operations are performed based on FIFO (First In First Out) 
* principle and the last position is connected back to the first position to make a circle.
* It is also called "Ring Buffer". One of the benefits of the circular queue is that
* we can make use of the spaces in front of the queue. */
class MyCircularQueue {
public:
	/** Initialize your data structure here. Set the size of the queue to be k. */
	MyCircularQueue(int k) {
		size = k; 
	}

	/** Insert an element into the circular queue. Return true if the operation is successful. */
	bool enQueue(int value) {
		if (isFull()) return false; 
		data.push_back(value);
		return true; 
	}

	/** Delete an element from the circular queue. Return true if the operation is successful. */
	bool deQueue() {
		if (isEmpty()) return false; 
		data.erase(data.begin());
		return true; 
	}

	/** Get the front item from the queue. */
	int Front() {
		if (isEmpty()) return -1; 
		return data.front(); 
	}

	/** Get the last item from the queue. */
	int Rear() {
		if (isEmpty()) return -1; 
		return data.back(); 
	}

	/** Checks whether the circular queue is empty or not. */
	bool isEmpty() {
		return data.empty(); 
	}

	/** Checks whether the circular queue is full or not. */
	bool isFull() {
		return data.size() >= size; 
	}

private:
	vector<int> data; 
	int size; 
};

/* 604. Design Compressed String Iterator */
/* Design and implement a data structure for a compressed string iterator. 
* It should support the following operations: next and hasNext.
* The given compressed string will be in the form of each letter followed 
* by a positive integer representing the number of this letter existing in 
* the original uncompressed string.
* next() - if the original string still has uncompressed characters, 
* return the next letter; Otherwise return a white space.
* hasNext() - Judge whether there is any letter needs to be uncompressed.
* StringIterator iterator = new StringIterator("L1e2t1C1o1d1e1"). */
class StringIterator {
public:
	StringIterator(string compressedString) {
		s = compressedString;
		cnt = 0;
		i = 0;
		n = s.size();
	}

	char next() {
		if (hasNext()) {
			--cnt;
			return c;
		}
		return ' ';
	}

	bool hasNext() {
		if (cnt) return true;
		if (i >= n) return false;
		c = s[i++];

		while (i < n && s[i] >= '0' && s[i] <= '9') {
			cnt = cnt * 10 + (s[i++] - '0');
		}
		return true;
	}

private:
	string s; 
	int i, n, cnt;
	char c; 
};

/* 706. Design HashMap */
/* Design a HashMap without using any built-in hash table libraries. 
* put(key, value) : Insert a (key, value) pair into the HashMap. 
* If the value already exists in the HashMap, update the value.
* get(key): Returns the value to which the specified key is mapped, 
* or -1 if this map contains no mapping for the key.
* remove(key) : Remove the mapping for the value key if this map contains 
* the mapping for the key. */
class MyHashMap {
	vector<list<pair<int, int>>> v;
	int m_size = 10000;
public:
	/** Initialize your data structure here. */
	MyHashMap() {
		v.resize(m_size);
	}

	/** value will always be non-negative. */
	void put(int key, int value) {
		auto &list = v[key % m_size];
		for (auto & val : list) {
			if (val.first == key) {
				val.second = value;
				return;
			}
		}
		list.emplace_back(key, value);
	}

	/** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
	int get(int key) {
		const auto list = v[key % m_size];
		if (list.empty()) {
			return -1;
		}
		for (auto val : list) {
			if (val.first == key) {
				return val.second;
			}
		}
		return -1;
	}

	/** Removes the mapping of the specified value key if this map contains a mapping for the key */
	void remove(int key) {
		auto &list = v[key % m_size];
		list.remove_if([key](auto n) { return n.first == key; });
	}
};

/* 362. Design Hit Counter */
/* Design a hit counter which counts the number of hits received in the past 5 minutes. */
class HitCounter {
public:
	/** Initialize your data structure here. */
	HitCounter() {

	}

	/** Record a hit.
	@param timestamp - The current timestamp (in seconds granularity). */
	void hit(int timestamp) {
		q.push(timestamp);
	}

	/** Return the number of hits in the past 5 minutes.
	@param timestamp - The current timestamp (in seconds granularity). */
	int getHits(int timestamp) {
		while (!q.empty() && timestamp - q.front() >= 300) {
			q.pop();
		}
		return q.size();
	}

private:
	queue<int> q;
};

/* 635. Design Log Storage System */
/* void Put(int id, string timestamp): Given a log's unique id and timestamp,
* store the log in your storage system.
* int[] Retrieve(String start, String end, String granularity): 
* Return the id of logs whose timestamps are within the range from start to end.*/
class LogSystem {
public:
	LogSystem() {
		units = { "Year", "Month", "Day", "Hour", "Minute", "Second" };
		positions = { 4, 7, 10, 13, 16, 19 };
	}

	void put(int id, string timestamp) {
		vec.push_back({ id, timestamp });
	}

	vector<int> retrieve(string s, string e, string gra) {
		vector<int> res;
		int i = positions[find(units.begin(), units.end(), gra) - units.begin()];

		for (auto a : vec) {
			auto t = a.second;
			if (t.substr(0, i).compare(s.substr(0, i)) >= 0 && t.substr(0, i).compare(e.substr(0, i)) <= 0) {
				res.push_back(a.first);
			}
		}

		return res;
	}

private:
	vector<pair<int, string>> vec;
	vector<string> units;
	vector<int> positions;
};

/* 353. Design Snake Game */
/* Design a Snake game that is played on a device with screen size = width x height. 
* Play the game online if you are not familiar with the game.
* The snake is initially positioned at the top left corner (0,0) with length = 1 unit.
* You are given a list of food's positions in row-column order. 
* When a snake eats the food, its length and the game's score both increase by 1.
* Each food appears one by one on the screen. For example, the second food will 
* not appear until the first food was eaten by the snake. When a food does appear 
* on the screen, it is guaranteed that it will not appear on a block occupied by the snake. */
class SnakeGame {
public:
	/** Initialize your data structure here.
	@param width - screen width
	@param height - screen height
	@param food - A list of food positions
	E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0]. */
	SnakeGame(int width, int height, vector<pair<int, int>>& food) {
		score = 0;
		this->width = width;
		this->height = height;
		this->food = food;
		snake.push_back({ 0, 0 });
	}

	/** Moves the snake.
	@param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
	@return The game's score after the move. Return -1 if game over.
	Game over when snake crosses the screen boundary or bites its body. */
	int move(string direction) {
		auto head = snake.front(), tail = snake.back();
		snake.pop_back();

		if (direction == "U") --head.first;
		else if (direction == "D") ++head.first;
		else if (direction == "L") --head.second;
		else if (direction == "R") ++head.second;

		if (head.first < 0 || head.first >= height || head.second < 0 || head.second >= width || count(snake.begin(), snake.end(), head)) return -1;

		snake.insert(snake.begin(), head);

		if (!food.empty() && head == food.front()) {
			++score;
			food.erase(food.begin());
			snake.push_back(tail);
		}

		return score;
	}

private:
	int score, width, height;
	vector<pair<int, int> > food, snake;
};

/* 348. Design Tic-Tac-Toe */
/* Design a Tic-tac-toe game that is played between two players on a n x n grid.
* You may assume the following rules:
* A move is guaranteed to be valid and is placed on an empty block.
* Once a winning condition is reached, no more moves is allowed.
* A player who succeeds in placing n of their marks in a horizontal, vertical, 
* or diagonal row wins the game. */
class TicTacToe {
public:
	/** Initialize your data structure here. */
	TicTacToe(int n) {
		board.resize(n, vector<int>(n, 0));
	}

	/** Player {player} makes a move at ({row}, {col}).
	@param row The row of the board.
	@param col The column of the board.
	@param player The player, can be either 1 or 2.
	@return The current winning condition, can be either:
	0: No one wins.
	1: Player 1 wins.
	2: Player 2 wins. */
	int move(int row, int col, int player) {
		board[row][col] = player;
		int n = board.size(), i = 0, j = 0;

		for (i = 1; i < n; ++i) {
			if (board[i][col] != board[i - 1][col]) break;
		}
		if (i == n) return player;

		for (j = 1; j < n; ++j) {
			if (board[row][j] != board[row][j - 1]) break;
		}
		if (j == n) return player;

		if (row == col) {
			for (i = 1; i < n; ++i) {
				if (board[i][i] != board[i - 1][i - 1]) break;
			}
			if (i == n) return player;
		}
		if (row + col == n - 1) {
			for (i = 1; i < n; ++i) {
				if (board[n - 1 - i][i] != board[n - i][i - 1]) break;
			}
			if (i == n) return player;
		}
		return 0;
	}

private:
	vector<vector<int> > board;
};

/* 271. Encode and Decode Strings */
/* Design an algorithm to encode a list of strings to a string. 
* The encoded string is then sent over the network and is decoded 
* back to the original list of strings. */
class Codec {
public:
	// Encodes a list of strings to a single string.
	string encode(vector<string>& strs) {
		string res("");
		for (auto s : strs) {
			res += to_string(s.size()) + "/" + s;
		}
		return res; 
	}

	// Decodes a single string to a list of strings.
	vector<string> decode(string s) {
		vector<string> res; 
		int i = 0, n = s.size();
		while (i < n) {
			int ix = s.find_first_of("/", i);
			int len = stoi(s.substr(i, ix - i));
			res.push_back(s.substr(ix + 1, len));
			i = ix + len + 1; 
		}
		return res; 
	}
};

/* 535. Encode and Decode TinyURL */
/* TinyURL is a URL shortening service where you enter a URL such as 
*  https: //leetcode.com/problems/design-tinyurl 
*  and it returns a short URL such as http: //tinyurl.com/4e9iAk. */
class EncodeDecodeTinyURL {
public:
	// Encodes a URL to a shortened URL.
	string encode(string longUrl) {
		v.push_back(longUrl);
		return "http: //tinyurl.com/" + to_string(v.size() - 1); 
	}

	// Decodes a shortened URL to its original URL.
	string decode(string shortUrl) {
		int i = shortUrl.find_last_of("/");
		int ix = stoi(shortUrl.substr(i + 1)); 
		return v[ix];
	}

private: 
	int i; 
	vector<string> v; 
};

/* 855. Exam Room */
/* In an exam room, there are N seats in a single row, numbered 0, 1, 2, ..., N-1.
* When a student enters the room, they must sit in the seat that maximizes 
* the distance to the closest person.  If there are multiple such seats, they sit 
* in the seat with the lowest number.  (Also, if no one is in the room,
* then the student sits at seat number 0.) Return a class ExamRoom(int N) that 
* exposes two functions. It is guaranteed that any calls to ExamRoom.leave(p) have
* a student sitting in seat p. */
class ExamRoom {
public:
	ExamRoom(int N) {
		n = N;
	}

	int seat() {
		int mx = 0, idx = 0, start = 0;
		for (auto i : positions) {
			if (start == 0) {
				if (mx < i - start) {
					mx = i - start;
					idx = 0;
				}
			}
			else {
				if (mx < (i - start + 1) / 2) {
					mx = (i - start + 1) / 2;
					idx = start + mx - 1;
				}
			}
			start = i + 1;
		}
		if (start > 0 && mx < n - start) {
			mx = n - start;
			idx = n - 1;
		}
		positions.insert(idx);
		return idx;
	}

	void leave(int p) {
		positions.erase(p);
	}

private:
	set<int> positions;
	int n;
};

/* 295. Find Median from Data Stream */
/* Median is the middle value in an ordered integer list. If the size of the list is even, 
* there is no middle value. So the median is the mean of the two middle value. 
* For example,[2,3,4], the median is 3. [2,3], the median is (2 + 3) / 2 = 2.5. */
class MedianFinder {
private:
	priority_queue<long> small, large;  //queue container is from large to small

public:
	// Adds a number into the data structure.
	void addNum(int num) {
		small.push(num);
		large.push(-small.top());
		small.pop();
		if (small.size() < large.size()) {
			small.push(-large.top());
			large.pop();
		}
	}

	// Returns the median of current data stream
	double findMedian() {
		return small.size() > large.size() ? small.top() : 0.5 *(small.top() - large.top());
	}
};

/* 676. Implement Magic Dictionary */
/* Implement a magic directory with buildDict, and search methods. For the method buildDict, 
* you'll be given a list of non-repetitive words to build a dictionary. */

class MagicDictionary {
public:
	/** Initialize your data structure here. */
	MagicDictionary() {

	}

	/** Build a dictionary through a list of words */
	void buildDict(vector<string> dict) {
		for (auto s : dict) {
			m[s.size()].push_back(s);
		}
	}

	/** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
	bool search(string word) {
		int len = word.size(); 
		for (auto s : m[len]) {
			int cnt = 0, i = 0; 
			for (i = 0; i < len; ++i) {
				if (word[i] == s[i]) continue; 
				if (word[i] != s[i] && cnt == 1) break; 
				++cnt;
			}
			if (i == len && cnt == 1) return true; 
		}
		return false; 
	}

private: 
	unordered_map<int, vector<string>> m;
};

/* 208. Implement Trie (Prefix Tree) Implement a trie with insert, search, and startsWith methods. */
class Trie {
public:
	struct TrieNode {
		TrieNode* child[26];
		bool isWord;
		TrieNode() : isWord(false) {
			for (auto &a : child) a = NULL;
		}
	};

	/** Initialize your data structure here. */
	Trie() {
		root = new TrieNode();
	}

	/** Inserts a word into the trie. */
	void insert(string word) {
		TrieNode* p = root;
		for (auto a : word) {
			int i = a - 'a';
			if (!p->child[i]) p->child[i] = new TrieNode();
			p = p->child[i];
		}
		p->isWord = true;
	}

	/** Returns if the word is in the trie. */
	bool search(string word) {
		TrieNode* p = root;
		for (auto a : word) {
			int i = a - 'a';
			if (!p->child[i]) return false;
			p = p->child[i];
		}
		return p->isWord;
	}

	/** Returns if there is any word in the trie that starts with the given prefix. */
	bool startsWith(string prefix) {
		TrieNode* p = root;
		for (auto a : prefix) {
			int i = a - 'a';
			if (!p->child[i]) return false;
			p = p->child[i];
		}
		return true;
	}

private:
	TrieNode* root;
};

/* 380. Insert Delete GetRandom O(1) */
/* Design a data structure that supports all following operations in average O(1) time.
* insert(val): Inserts an item val to the set if not already present.
* remove(val): Removes an item val from the set if present.
* getRandom: Returns a random element from current set of elements. 
* Each element must have the same probability of being returned. */
class RandomizedSet {
public:
	/** Initialize your data structure here. */
	RandomizedSet() {

	}

	/** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
	bool insert(int val) {
		if (m.count(val)) return false;
		nums.push_back(val);
		m[val] = nums.size() - 1;
		return true;
	}

	/** Removes a value from the set. Returns true if the set contained the specified element. */
	/* O(1) Time complexity. swap the last element with the one need to be deleted. Then pop the
	 * one from the back. */
	bool remove(int val) {
		if (!m.count(val)) return false;
		int last = nums.back();
		m[last] = m[val];
		nums[m[val]] = last;
		nums.pop_back();
		m.erase(val);
		return true;
	}

	/** Get a random element from the set. */
	int getRandom() {
		return nums[rand() % nums.size()];
	}

private:
	vector<int> nums;
	unordered_map<int, int> m;
};

/* 146. LRU Cache */
/* Design and implement a data structure for Least Recently Used (LRU) cache. 
* It should support the following operations: get and put.
* get(key) - Get the value (will always be positive) of the key if the key exists in the cache, 
* otherwise return -1.
* put(key, value) - Set or insert the value if the key is not already present. 
* When the cache reached its capacity, it should invalidate the least recently used item before 
*& inserting a new item. */
class LRUCache {
public:
	LRUCache(int capacity) {
		cap = capacity;
	}

	int get(int key) {
		auto it = m.find(key);
		if (it == m.end()) return -1;
		l.splice(l.begin(), l, it->second);
		return it->second->second;
	}

	void put(int key, int value) {
		auto it = m.find(key);
		if (it != m.end()) l.erase(it->second);
		l.push_front({ key, value });
		m[key] = l.begin();

		if (m.size() > cap) {
			int k = l.rbegin()->first;
			l.pop_back();
			m.erase(k);
		}
	}

private:
	int cap;
	list<pair<int, int>> l;
	unordered_map<int, list<pair<int, int>> ::iterator> m;
};

/* 382. Linked List Random Node */
/* Given a singly linked list, return a random node's value from the linked list. 
* Each node must have the same probability of being chosen. */
class Solution {
public:
	/** @param head The linked list's head.
	Note that the head is guaranteed to be not null, so it contains at least one node. */
	Solution(ListNode* head) {
		this->head = head; 
		int len = 0; 
		while (head) {
			++len; 
			head = head->next; 
		}
	}

	/** Returns a random node's value. */
	int getRandom() {
		int t = rand() % len; 
		this->head = head; 
		ListNode* cur = head; 
		while (t > 0) {
			cur = cur->next; 
			--t; 
		}
		return cur->val; 
	}
private: 
	int len; 
	ListNode* head; 
};

/* 359. Logger Rate Limiter */
class Logger {
public:
	/** Initialize your data structure here. */
	Logger() {

	}
	/** Returns true if the message should be printed in the given timestamp, otherwise returns false.
	If this method returns false, the message will not be printed.
	The timestamp is in seconds granularity. */
	bool shouldPrintMessage(int timestamp, string message) {
		if (m.find(message) != m.end()) {
			if (timestamp - m[message] >= 10) {
				m[message] = timestamp;
				return true;
			}
		}
		else {
			m[message] = timestamp;
			return true;
		}
		return false;
	}
private:
	unordered_map<string, int> m;
};

/* 745. Prefix and Suffix Search */
/* Given many words, words[i] has weight i. Design a class WordFilter that supports one function, 
* WordFilter.f(String prefix, String suffix). It will return the word with given prefix and suffix 
* with maximum weight. If no word exists, return -1. */
class WordFilter {
public:
	WordFilter(vector<string>& words) {
		for (int k = 0; k < words.size(); ++k) {
			for (int i = 0; i <= words[k].size(); ++i) {
				mpre[words[k].substr(0, i)].push_back(k);
			}

			for (int j = 0; j <= words[k].size(); ++j) {
				msuf[words[k].substr(words[k].size() - j)].push_back(k);
			}
		}
	}

	int f(string prefix, string suffix) {
		if (!mpre.count(prefix) || !msuf.count(suffix)) return -1;
		vector<int> pre = mpre[prefix], suf = msuf[suffix];
		int i = pre.size() - 1, j = suf.size() - 1;

		while (i >= 0 && j >= 0) {
			if (pre[i] > suf[j]) --i;
			else if (pre[i] < suf[j]) --j;
			else return pre[i];
		}
		return -1;
	}

private:
	unordered_map<string, vector<int>> mpre, msuf;
};

/* 528. Random Pick with Weight */
/* Given an array w of positive integers, where w[i] describes the weight of index i, 
* write a function pickIndex which randomly picks an index in proportion to its weight.
* w = [1, 3, 2] -> sums = [1, 4, 6] 
* 比如若权重数组为 [1, 3, 2] 的话，那么累加和数组为 [1, 4, 6]，整个的权重和为6，我们 rand() % 6，
* 可以随机出范围 [0, 5] 内的数， */
class Solution {
public:
	Solution(vector<int>& w) {
		sums = w; 
		for (int i = 1; i < w.size(); ++i) {
			sums[i] += sums[i - 1];
		}
	}

	int pickIndex() {
		int t = rand() % sums.back(), left = 0, right = sums.size() - 1; 
		while (left < right) {
			int mid = left + (right - left) / 2; 
			if (t < sums[mid]) right = mid; 
			else left = mid + 1; 
		}
		return right; 
	}

private:
	vector<int> sums; 
};

/* 497. Random Point in Non-overlapping Rectangles */
/* Given a list of non-overlapping axis-aligned rectangles rects, write a function pick which randomly 
 * and uniformily picks an integer point in the space covered by the rectangles. */
class Solution {
public:
	Solution(vector<vector<int>>& rects) {
		this->rects = rects;
	}
	vector<int> pick() {
		int sum = 0;
		vector<int> selected;
		/* Step 1 - select a random rectangle considering the area of it. */
		for (auto r : rects) {
			/* What we need to be aware of here is that the input may contain
			 * lines that are not rectangles. For example, [1, 2, 1, 5], [3, 2, 3, -2].
			 * So, we work around it by adding +1 here. It does not affect
			 * the final result of reservoir sampling. */
			int area = (r[2] - r[0] + 1) * (r[3] - r[1] + 1);
			sum += area;
			if (rand() % sum < area) selected = r;
		}
		/* Step 2 - select a random (x, y) coordinate within the selected rectangle. */
		int x = rand() % (selected[2] - selected[0] + 1) + selected[0];
		int y = rand() % (selected[3] - selected[1] + 1) + selected[1];
		return { x, y };
	}

private:
	vector<vector<int>> rects;
	vector<int> selected;
};

/* 449. Serialize and Deserialize BST */
/* Design an algorithm to serialize and deserialize a binary search tree. There is no restriction on how your 
* serialization/deserialization algorithm should work. You just need to ensure that a binary search tree 
* can be serialized to a string and this string can be deserialized to the original tree structure. */
/* 297. Serialize and Deserialize Binary Tree */
/* Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your
* serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be
* serialized to a string and this string can be deserialized to the original tree structure. */
class Codec {
public:

	void serialize_helper(TreeNode* root, ostringstream& os) {
		if (!root) os << "# ";
		else {
			os << to_string(root->val) + " ";
			serialize_helper(root->left, os);
			serialize_helper(root->right, os);
		}
	}

	// Encodes a tree to a single string.
	string serialize(TreeNode* root) {
		ostringstream os;
		serialize_helper(root, os);
		return os.str();
	}

	TreeNode* deserialize_helper(istringstream& is) {
		string val("");
		is >> val;
		if (val == "#") return NULL;
		TreeNode* root = new TreeNode(stoi(val));
		root->left = deserialize_helper(is);
		root->right = deserialize_helper(is);
		return root;
	}

	// Decodes your encoded data to tree.
	TreeNode* deserialize(string data) {
		istringstream is(data);
		return deserialize_helper(is);
	}
};

/* 428. Serialize and Deserialize N-ary Tree */
/* Design an algorithm to serialize and deserialize an N-ary tree. An N-ary tree is a rooted tree in which 
* each node has no more than N children. There is no restriction on how your serialization/deserialization
* algorithm should work. You just need to ensure that an N-ary tree can be serialized to a string and this
* string can be deserialized to the original tree structure. */
class Codec2 {
public:
	void serializeHelper(NaryTreeNode* root, ostringstream& res) {
		if (!root) res << "# ";
		else {
			res << to_string(root->val) + " " + to_string(root->children.size()) + " ";
			for (auto child : root->children) {
				serializeHelper(child, res);
			}
		}
	}

	// Encodes a tree to a single string.
	string serialize(NaryTreeNode* root) {
		ostringstream res;
		serializeHelper(root, res);
		return res.str();
	}

	NaryTreeNode* deserializeHelper(istringstream& iss) {
		string val(""), size("");
		iss >> val;
		if (val == "#") return NULL;

		iss >> size;
		NaryTreeNode* root = new NaryTreeNode({ stoi(val),{} });

		for (int i = 0; i < stoi(size); ++i) {
			root->children.push_back(deserializeHelper(iss));
		}
		return root;
	}

	// Decodes your encoded data to tree.
	NaryTreeNode* deserialize(string data) {
		istringstream iss(data);
		return deserializeHelper(iss);
	}
};

/* 384. Shuffle an Array */
/* Shuffle a set of numbers without duplicates. */
class ShuffleArray {
public:
	ShuffleArray(vector<int>& nums) : vec(nums) {

	}
	/** Resets the array to its original configuration and return it. */
	vector<int> reset() {
		return vec;
	}
	/** Returns a random shuffling of the array. */
	vector<int> shuffle() {
		vector<int> res(vec);
		int n = vec.size();
		for (int i = 0; i < n; ++i) {
			int idx = rand() % n;
			swap(res[i], res[idx]);
		}
		return res;
	}
private:
	vector<int> vec;
};



#endif

