#pragma once
#include<iostream>
#include<vector>
#include<list>
#include<queue>
#include<unordered_map>
using namespace std;


#ifndef BBG_HEADER_H
#define BBG_HEADER_H

class HitCounter {
private:
	queue<int> q; 
public: 
	HitCounter();
	void hit(int timestamp);
	int getHits(int timestamp);
}; 

struct ListNode {
	int val; 
	ListNode* next;
	ListNode();
	ListNode(int x);
};

struct TreeNode {
	int val; 
	TreeNode* left; 
	TreeNode* right; 
	TreeNode(int x);
};

struct DoubledLinkedList {
	int val; 
	DoubledLinkedList* prev;
	DoubledLinkedList* next; 
	DoubledLinkedList* child; 
	DoubledLinkedList(); 
	DoubledLinkedList(int x);
};

class RandomizedSet {
private:
	unordered_map<int, int> m; // save number and it's index in vec
	vector<int> vec; 

public:
	RandomizedSet(vector<int>& vec);
	bool insert(int val);
	bool remove(int val);
	int getRandom();
	void printSet();
};

class LRUCache {
private:
	int cap;
	list<pair<int, int>> l; 
	unordered_map<int, list<pair<int, int>>::iterator> m;

public:
	LRUCache(int capability);
	int get(int key);
	void put(int key, int value);
};


struct Interval {
	int start, end; 
	Interval(); 
	Interval(int s, int e);
};

class MinStack {
private:
	stack<int> st1, st2; 
public:
	MinStack(); 
	void push(int x); // Push element x onto stack.
	void pop(); // Removes the element on top of the stack.
	int top(); // Get the top element.
	int getMin(); // Retrieve the minimum element in the stack.
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

#endif

