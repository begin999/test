#include<iostream>
#include<vector>
#include<stack>
#include<string>
#include<deque>
#include<map>
#include<set>
#include<sstream>
#include<functional> // "greater<int>()"
#include<unordered_map>
#include<unordered_set>
#include"bbg_header.h"
using namespace std;


/* =============================================================== */
/* Todo list:
 * 1. Integer to English words 
 * 2. LFU Cache 
 * 889, 642, 390, 295, 510, 273, 12, 703, 215, 14, 516, 978, 716, 646, 
 * 682, 155, 346, 556, 933, 131, 60, 303, 158, 1029, 189, 48, 737, 71, 
 * 8, 811, 669, 125, 678, 
 */


 /* =============================================================== */
/* 教训：
 * 1. vector need to be initialized before doing 'res[i] = a;'
 * 
 * 2. difference between 'm.count(k)' and 'm[k]': 
 *    (1) 'm.count(k)' -- Searches the container for elements whose key is k and returns the number of elements found. 
 *         Because 'unordered_map' containers do not allow for duplicate keys, this means that the function actually 
 *		   returns 1 if an element with that key exists in the container, and zero otherwise.
 *    (2) 'm[k]' -- If k matches the key of an element in the container, the function returns a reference to its mapped value.
 * 
 * 3. Important to understand when to use "||" "&&"
 * 
 * 4. Linked list 循环条件， 不确定的话就过一两个例子仔细确定。
 * 
 * 5. STL::String. difference among: "+=", "push_back" and "append"
 *	  (1)  += : Doesn’t allow appending part of string.
 *	  (2)  append() : Allows appending part of string.
 *	  (3)  push_back : We can’t append part of string using push_back.
 */


 /* =============================================================== */
// All helper functions
vector<vector<int>> dirs{ { -1, 0 },{ 1, 0 },{ 0, -1 },{ 0, 1 } };
vector<vector<int>> dirs2{ { -2, -1 },{ -2, 1 },{ -1, 2 },{ 1, 2 },{ 2, 1 },
{ 2, -1 },{ 1, -2 },{ -1, -2 } };
vector<vector<int>> dirs3{ { -1, -1 },{ -1, 0 },{ -1, 1 },{ 0, -1 },{ 0, 1 },
{ 1, -1 },{ 1, 0 },{ 1, 1 } };

template <class T> 
void printVector(vector<T>& vec) {
	for (int i = 0; i < vec.size(); ++i) {
		cout << vec[i] << " ";
	}
	cout << endl;
}

// =============================================================================================
// All defined class implementations
/* ListNode class*/
ListNode::ListNode() {}
ListNode::ListNode(int x) : val(x) {}

/* TreeNode class*/
//TreeNode::TreeNode() {}
TreeNode::TreeNode(int x) : val(x), left(NULL), right (NULL) {}

/* Double linked list */
DoubledLinkedList::DoubledLinkedList() {}
DoubledLinkedList::DoubledLinkedList(int x) : val(x), prev(NULL), next(NULL), child(NULL) {}

/* Interval class */
Interval::Interval() {}
Interval::Interval(int s, int e) : start(s), end(e) {}

/* MinStack class */
MinStack::MinStack() {}

// Push element x onto stack.
void MinStack::push(int x) {
	st1.push(x);
	if (st2.empty() || st2.top() <= x) st2.push(x);
}

// Removes the element on top of the stack.
void MinStack::pop() {
	if (!st2.empty() && st2.top() == st1.top()) st2.pop();
	st1.pop(); 
}

// Get the top element.
int MinStack::top() {
	return st1.top(); 
}

// Retrieve the minimum element in the stack.
int MinStack::getMin() {
	return st2.top(); 
}


/* ======================== 1. Dynamic programming ================== */
/* 1. 650. 2 Keys Keyboard */
/* Initially on a notepad only one character 'A' is present.
* You can perform two operations on this notepad for each step:
* Copy All: You can copy all the characters present on the notepad
* (partial copy is not allowed). Paste: You can paste the characters
* which are copied last time. Given a number n. You have to get
* exactly n 'A' on the notepad by performing the minimum number
* of steps permitted. Output the minimum number of steps to get n 'A'. */
int minSteps(int n) {
	if (n == 1) return 0;
	int res = n;
	for (int i = 2; i <= n; ++i) {
		if (n % i == 0) {
			res = min(res, i + minSteps(n / i));
		}
	}
	return res;
}

/* 2. 413. Arithmetic Slices */
/* A sequence of number is called arithmetic if it consists of
* at least three elements and if the difference between any
* two consecutive elements is the same.
* The function should return the number of arithmetic slices
* in the array A.
* Example: A = [1, 2, 3, 4]. return: 3.
* A: [1, 2, 3], [2, 3, 4] and [1, 2, 3, 4] itself. */
int numberOfArithmeticSlices(vector<int>& A) {
	int res = 0, n = A.size();
	vector<int> dp(n, 0);
	for (int i = 2; i < n; ++i) {
		if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
			dp[i] = 1 + dp[i - 1];
			res += dp[i];
		}
	}
	return res;
}

/* 3. 446. Arithmetic Slices II - Subsequence */
/* A sequence of numbers is called arithmetic if it consists of at least three elements 
* and if the difference between ANY two consecutive elements is the same.
* Input: [2, 4, 6, 8, 10]. Output: 7. Explanation: All arithmetic subsequence slices are:
* [2,4,6], [4,6,8], [6,8,10], [2,4,6,8], [4,6,8,10], [2,4,6,8,10], [2,6,10]. */
int numberOfArithmeticSlices2(vector<int>& A) {
	int res = 0, n = A.size();
	vector<unordered_map<int, int>> m(n); // vector of {index, diff} pairs

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < i; ++j) { // IMPORTANT:  for each number, try from the beginning
			long long diff = (long long) A[i] - A[j];
			int t = (int)diff;
			if (diff > INT_MAX || diff < INT_MIN) continue;
			++m[i][t];

			if (m[j].count(t)) {
				res += m[j][t];
				m[i][t] += m[j][t];
			}
		}
	}
	return res;
}

/* 4. 70. Climbing Stairs */
/* You are climbing a stair case. It takes n steps to reach to the top. Each time you can 
* either climb 1 or 2 steps. In how many distinct ways can you climb to the top? */
int climbStairs(int n) {
	vector<int> dp(n + 1, 0); // n + 1
	dp[0] = 1, dp[1] = 1; // initialization dp[0] = 1; 
	for (int i = 2; i <= n; ++i) {
		dp[i] = dp[i - 1] + dp[i - 2];
	}
	return dp.back();
}

/* 5. 91. Decode ways */
/* A message containing letters from A-Z is being encoded to
* numbers using the following mapping: 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26
* Given a non-empty string containing only digits, determine
* the total number of ways to decode it. Input: "226". Output: 3. */
int numDecodings(string s) {
	int n = s.size();
	vector<int> dp(n + 1, 0);
	dp[0] = 1;

	for (int i = 1; i <= n; ++i) {
		dp[i] = s[i - 1] == '0' ? 0 : dp[i - 1];
		if (i > 1 && (s[i - 2] == '1' || (s[i - 2] == '2' && s[i - 1] <= '6'))) dp[i] += dp[i - 2];
	}
	return dp.back();
}

/* 6. 322. Coin change */
/* You are given coins of different denominations and a total amount
* of money amount. Write a function to compute the FEWEST number
* of coins that you need to make up that amount. If that amount
* of money cannot be made up by any combination of the coins, return -1.
* Input: coins = [1, 2, 5], amount = 11. Output: 3. Explanation: 11 = 5 + 5 + 1. */
int coinChange(vector<int>& coins, int amount) {
	vector<int> dp(amount + 1, amount + 1);
	dp[0] = 0; // initialization
	for (int i = 1; i <= amount; ++i) {
		for (auto a : coins) {
			if (i >= a) dp[i] = min(dp[i], 1 + dp[i - a]);
		}
	}
	// IMPORTANT.
	return dp.back() == amount + 1 ? -1 : dp.back();
}

/* 7. 518. Coin Change 2 */
/* You are given coins of different denominations and a total amount of money.
* Write a function to compute the number of combinations that make up that amount.
* You may assume that you have infinite number of each kind of coin.
* Input: amount = 5, coins = [1, 2, 5]. Output: 4.
* 5=5, 5=2+2+1, 5=2+1+1+1, 5=1+1+1+1+1.
* dp[i][j] 表示用前i个硬币组成钱数为j的不同组合方法，怎么算才不会重复，也不会漏掉呢？
* 我们采用的方法是一个硬币一个硬币的增加，每增加一个硬币，都从1遍历到 amount，对于遍历到的当前钱数j，
* 组成方法就是不加上当前硬币的拼法 dp[i-1][j]，还要加上，去掉当前硬币值的钱数的组成方法. */
int coinChange2(int amount, vector<int>& coins) {
	int n = coins.size();
	vector<vector<int>> dp(n + 1, vector<int>(amount + 1));
	dp[0][0] = 1;
	for (int i = 1; i <= n; ++i) {
		dp[i][0] = 1;
		for (int j = 1; j <= amount; ++j) {
			dp[i][j] = dp[i - 1][j] + (j >= coins[i - 1] ? dp[i][j - coins[i - 1]] : 0);
		}
	}
	return dp.back().back();
}

/* 8. 1105. Filling Bookcase Shelves -- CLASSIC DP PROBLEM */
/* We have a sequence of books: the i-th book has thickness
* books[i][0] and height books[i][1]. We want to place these
* books in order onto bookcase shelves that have total width
* shelf_width. Return the MINIMUM possible height that the total
* bookshelf can be after placing shelves in this manner.
* Constraints:
* 1 <= books.length <= 1000
* 1 <= books[i][0] <= shelf_width <= 1000
* 1 <= books[i][1] <= 1000.
* Define: dp[i] is the min height after placing ith book. */


int minHeightShelves(vector<vector<int>>& books, int shelf_width) {
	int n = books.size();
	vector<int> dp(n + 1, 1000 * 1000);
	dp[0] = 0;

	for (int i = 1; i <= n; ++i) {
		auto b = books[i - 1];
		int w = b[0], h = b[1];
		// if let current book be on its own row
		dp[i] = dp[i - 1] + h;
		// if putting with previous books
		for (int j = i - 1; j > 0; --j) {
			w += books[j - 1][0];
			if (w > shelf_width) break;
			h = max(h, books[j - 1][1]);
			dp[i] = min(dp[i], dp[j - 1] + h);
		}
	}
	return dp.back();
}

/* 5. Longest Palindromic Substring */
/* Given a string s, find the longest palindromic substring in s.
* You may assume that the maximum length of s is 1000.*/
string longestPalindrome2(string s) {
	if (s.empty()) return "";
	int n = s.size(), left = 0, right = 0, len = 0;
	vector<vector<int>> dp(n, vector<int>(n, 0));

	for (int i = n - 1; i >= 0; --i) {
		dp[i][i] = 1;
		for (int j = i + 1; j < n; ++j) {
			dp[i][j] = s[i] == s[j] && (j - i < 2 || dp[i + 1][j - 1]);
			if (dp[i][j] && len < j - i + 1) {
				len = j - i + 1;
				left = i;
				right = j;
			}
		}
	}
	return s.substr(left, right - left + 1);
}

/* 10. Regular Expression Matching */
/* Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
* '.' Matches any single character. '*' Matches zero or more of the preceding element. The matching should
* cover the entire input string (not partial).
* Note: s could be empty and contains only lowercase letters a-z.
* p could be empty and contains only lowercase letters a-z, and characters like . or *. */
bool isMatch(string s, string p) {
	int m = s.size(), n = p.size();
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	dp[0][0] = 1;
	for (int i = 0; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (j > 1 && p[j - 1] == '*') {
				dp[i][j] = (i > 0 && dp[i - 1][j] && (s[i - 1] == p[j - 2] || p[j - 2] == '.')) || dp[i][j - 2];
			}
			else {
				dp[i][j] = i > 0 && dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
			}
		}
	}
	return dp.back().back();
}

/* 161. One Edit Distance */
/* Given two strings s and t, determine if they are both one edit distance apart.
* Note: There are 3 possiblities to satisify one edit distance apart:
* Insert a character into s to get t
* Delete a character from s to get t
* Replace a character of s to get t
* Example 1: Input: s = "ab", t = "acb". Output: true. */
bool isOneEditDistance(string s, string t) {
	int m = s.size(), n = t.size(), diff = abs(m - n);
	if (m > n) swap(s, t);

	if (diff > 1) {
		return false;
	}
	else if (diff == 1) {
		for (int i = 0; i < n; ++i) {
			if (s[i] != t[i]) return s.substr(i) == t.substr(i + 1);
		}
	}
	else if (diff == 0) {
		int cnt = 0;
		for (int i = 0; i < n; ++i) {
			if (s[i] != t[i]) ++cnt;
		}
		return cnt == 1;
	}
	return true;
}

/* 97. Interleaving String */
/* Given s1, s2, s3, find whether s3 is formed by the
* interleaving of s1 and s2. Example 1:
* Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
* Output: true. */
bool isInterleave(string s1, string s2, string s3) {
	int m = s1.size(), n = s2.size();
	if (s3.size() != m + n) return false;
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	dp[0][0] = 1;
	for (int i = 1; i <= m; ++i) dp[i][0] = s1[i - 1] == s3[i - 1] && dp[i - 1][0];
	for (int j = 1; j <= n; ++j) dp[0][j] = s2[j - 1] == s3[j - 1] && dp[0][j - 1];
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			dp[i][j] = (s1[i - 1] == s3[i - 1 + j] && dp[i - 1][j]) ||
				(s2[j - 1] == s3[j - 1 + i] && dp[i][j - 1]);
		}
	}
	return dp.back().back();
}

/* 121. Best Time to Buy and Sell Stock */
/* Say you have an array for which the ith element is the price
* of a given stock on day i. If you were only permitted to
* complete at most one transaction (i.e., buy one and sell
* one share of the stock), design an algorithm to find the maximum
* profit. Note that you cannot sell a stock before you buy one. */
int maxProfit(vector<int>& prices) {
	int mn = INT_MAX, res = 0;
	for (auto a : prices) {
		mn = min(mn, a);
		res = max(res, a - mn);
	}
	return res;
}

/* 123. Best Time to Buy and Sell Stock III, IV */
/* Say you have an array for which the ith element is the price of
* a given stock on day i. Design an algorithm to find the maximum profit.
* You may complete AT MOST TWO transactions. Note: You may not engage in
* multiple transactions at the same time (i.e., you must sell the stock
* before you buy again).
* global[i][j]: 一个是当前到达第i天可以最多进行j次交易，最好的利润是多少.
* local[i][j]: 当前到达第i天，最多可进行j次交易，并且最后一次交易在当天卖出的最好的利润是多少.*/
int maxProfit34(vector<int>& prices) {
	if (prices.empty()) return 0;
	int k = 2, n = prices.size();
	vector<int> local(k + 1, 0), global(k + 1, 0);

	for (int i = 0; i < n - 1; ++i) {
		int diff = prices[i + 1] - prices[i];
		for (int j = k; j >= 1; --j) {
			local[j] = max(global[j - 1] + max(diff, 0), local[j] + diff);
			global[j] = max(global[j], local[j]);
		}
	}
	return global.back();
}

/* 309. Best Time to Buy and Sell Stock with Cooldown */
/* Say you have an array for which the ith element is the price of a given stock on day i.
* Design an algorithm to find the maximum profit. You may complete as many transactions
* as you like (ie, buy one and sell one share of the stock multiple times) with the
* following restrictions: You may not engage in multiple transactions at the same time
* (ie, you must sell the stock before you buy again). After you sell your stock,
* you cannot buy stock on next day. (ie, cooldown 1 day)
* Input: [1,2,3,0,2]. Output:3. Explanation: transactions = [buy,sell,cooldown,buy,sell].
* buy[i]表示在第i天之前最后一个操作是买，此时的最大收益。
* sell[i]表示在第i天之前最后一个操作是卖，此时的最大收益。
* rest[i]表示在第i天之前最后一个操作是冷冻期，此时的最大收益。
* buy[i]  = max(rest[i-1] - price, buy[i-1])
* sell[i] = max(buy[i-1] + price, sell[i-1])
* rest[i] = max(sell[i-1], buy[i-1], rest[i-1])
* 由于冷冻期的存在，可以得出rest[i] = sell[i-1]，这样可以将上面三个递推式精简到两个：
* buy[i]  = max(sell[i-2] - price, buy[i-1])
* sell[i] = max(buy[i-1] + price, sell[i-1])*/
int maxProfitCooldown(vector<int>& prices) {
	int pre_buy = 0, buy = INT_MIN, pre_sell = 0, sell = 0;
	for (auto price : prices) {
		pre_buy = buy;
		buy = max(pre_buy, pre_sell - price);
		pre_sell = sell;
		sell = max(pre_sell, pre_buy + price);
	}
	return sell;
}

/* 714. Best Time to Buy and Sell Stock with Transaction Fee  */
/* Your are given an array of integers prices, for which the i-th element is the
* price of a given stock on day i; and a non-negative integer fee representing a
* transaction fee. You may complete as many transactions as you like, but you
* need to pay the transaction fee for each transaction. You may not buy more than
* 1 share of a stock at a time (ie. you must sell the stock share before you buy again.)
* Return the maximum profit you can make.
* Input: prices = [1, 3, 2, 8, 4, 9], fee = 2. Output: 8
* sold[i]表示第i天卖掉股票此时的最大利润，hold[i]表示第i天保留手里的股票此时的最大利润。
* sold[i] = max(sold[i - 1], hold[i - 1] + price[i] - fee)
* hold[i] = max(hold[i - 1], sold[i - 1] - price[i]) */
int maxProfitFee(vector<int>& prices, int fee) {
	int n = prices.size();
	vector<int> sold(n, 0), hold(sold);
	hold[0] = -prices[0];
	for (int i = 1; i < n; ++i) {
		sold[i] = max(sold[i - 1], hold[i - 1] + prices[i] - fee);
		hold[i] = max(hold[i - 1], sold[i - 1] - prices[i]);
	}
	return sold.back();
}

/* 198. House Robber */
/* Given a list of non-negative integers representing the amount of
* money of each house, determine the maximum amount of money you
* can rob tonight without alerting the police.*/
int rob(vector<int>& nums) {
	if (nums.size() <= 1) return nums.size() == 0 ? 0 : nums[0];
	int n = nums.size();
	vector<int> dp(n, 0);
	dp[0] = nums[0], dp[1] = max(nums[0], nums[1]);
	for (int i = 2; i < n; ++i) {
		dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
	}
	return dp.back();
}

/* 213. House Robber II */
/* All houses at this place are arranged in a circle. Given a list
* of non-negative integers representing the amount of money of each house,
* determine the maximum amount of money you can rob tonight without
* alerting the police.*/
int robCircle(vector<int>& nums, int left, int right) {
	if (left > right) return 0;
	vector<int> dp(right + 1, 1);
	dp[left] = nums[left], dp[left + 1] = max(nums[left], nums[left + 1]);

	for (int i = left + 2; i <= right; ++i) {
		dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
	}
	return dp.back();
}

int robCircle(vector<int>& nums) {
	int n = nums.size();
	if (n <= 1) return nums.empty() ? 0 : nums[0];
	if (n == 2) return max(nums[0], nums[1]);
	return max(robCircle(nums, 0, n - 2), robCircle(nums, 1, n - 1));
}

/* 329. Longest Increasing Path in a Matrix */
/* Given an integer matrix, find the length of the longest increasing path.
* From each cell, you can either move to four directions:
* left, right, up or down. You may NOT move diagonally or
* move outside of the boundary (i.e. wrap-around is not allowed).
* Input: nums =
[[9,9,4],
[6,6,8],
[2,1,1]] . Output: 4. */
int longestIncreasingPath(vector<vector<int>>& matrix, vector<vector<int>>& dp, int i, int j) {
	int res = 1, m = matrix.size(), n = matrix[0].size();
	if (dp[i][j]) return dp[i][j];

	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		if (x >= 0 && x < m && y >= 0 && y < n && matrix[x][y] >= 1 + matrix[i][j]) {
			res = max(res, 1 + longestIncreasingPath(matrix, dp, x, y));
		}
	}
	return dp[i][j] = res;
}

int longestIncreasingPath(vector<vector<int>>& matrix) {
	int m = matrix.size(), n = matrix[0].size(), res = 1;
	vector<vector<int>> dp(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			res = max(res, longestIncreasingPath(matrix, dp, i, j));
		}
	}
	return res;
}

/* 221. Maximal Square */
/* Given a 2D binary matrix filled with 0's and 1's, find the largest
* square containing only 1's and return its area. Example: Input:
* 1 0 1 0 0
* 1 0 1 1 1
* 1 1 1 1 1
* 1 0 0 1 0. Output: 4 */
int maximalSquare(vector<vector<char>>& matrix) {
	if (matrix.empty() || matrix[0].empty()) return 0;
	int m = matrix.size(), n = matrix[0].size(), res = 0;
	vector<vector<int>> dp(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			// IMPORTANT: initizalization
			if (i == 0 || j == 0) dp[i][j] = matrix[i][j] == '0' ? 0 : 1;
			else if (matrix[i][j] == '1') {
				dp[i][j] = 1 + min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1]));
			}
			res = max(res, dp[i][j]);
		}
	}
	return res * res;
}

/* 64. Minimum Path Sum */
/* Given a m x n grid filled with non-negative numbers, find a path from
* top left to bottom right which minimizes the sum of all numbers along
* its path.Note:You can only move either down or right at any point in time. */
int minPathSum(vector<vector<int>>& grid) {
	if (grid.empty() || grid[0].empty()) return 0;
	int m = grid.size(), n = grid[0].size();
	vector<vector<int> > dp(m, vector<int>(n, 0));
	dp[0][0] = grid[0][0];
	for (int i = 1; i < m; ++i) dp[i][0] = dp[i - 1][0] + grid[i][0];
	for (int j = 1; j < n; ++j) dp[0][j] = dp[0][j - 1] + grid[0][j];

	for (int i = 1; i < m; ++i) {
		for (int j = 1; j < n; ++j) {
			dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
		}
	}
	return dp.back().back();
}

/* 727. Minimum Window Subsequence */
/* Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.
* If there is no such window in S that covers all characters in T, return the empty string "".
* If there are multiple such minimum-length windows, return the one with the left-most starting index.
* Input:  S = "abcdebdde", T = "bde". Output: "bcde". */
string minWindow(string S, string T) {
	int m = S.size(), n = T.size(), start = -1, minLen = INT_MAX;

	vector<vector<int>> dp(m + 1, vector<int>(n + 1, -1));
	for (int i = 0; i <= m; ++i) dp[i][0] = i;
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			dp[i][j] = (S[i - 1] == T[j - 1]) ? dp[i - 1][j - 1] : dp[i - 1][j];
		}
		if (dp[i].back() != -1) {
			int len = i - dp[i].back();
			if (minLen > len) {
				minLen = i - dp[i].back();
				start = dp[i].back();
			}
		}
	}
	return (start == -1) ? "" : S.substr(start, minLen);
}

/* 76. Minimum Window Substring -- sliding window & map */
/* Given a string S and a string T, find the minimum window in S which will contain all the
* characters in T in complexity O(n). Example: Input: S = "ADOBECODEBANC", T = "ABC". Output: "BANC". */
string minWindow(string s, string t) {
	string res("");
	unordered_map<char, int> m;
	for (auto c : t) ++m[c];
	int left = 0, len = INT_MAX, k = t.size(), cnt = 0;

	for (int i = 0; i < s.size(); ++i) {
		if (--m[s[i]] >= 0) ++cnt;

		while (cnt >= k) {
			if (len > i - left + 1) {
				len = i - left + 1;
				res = s.substr(left, len);
			}
			if (++m[s[left]] > 0) --cnt;
			++left;
		}
	}
	return res;
}

/* 132. Palindrome Partitioning II */
/* Given a string s, partition s such that every substring of the partition
* is a palindrome. Return the minimum cuts needed for a palindrome
* partitioning of s.Input: "aab". Output: 1. */
int minCut(string s) {
	int n = s.size();
	vector<vector<int> > P(n, vector<int>(n, 0));
	vector<int> dp(n + 1, 0);
	for (int i = 0; i <= n; ++i) dp[i] = n - i - 1;

	for (int i = n - 1; i >= 0; --i) {
		P[i][i] = 1;
		for (int j = i; j < n; ++j) {
			P[i][j] = s[i] == s[j] && (j - i < 2 || P[i + 1][j - 1]);
			if (P[i][j]) {
				dp[i] = min(dp[i], 1 + dp[j + 1]); // IMPORTANT:"dp[j + 1]".
			}
		}
	}
	return dp[0];
}

/* 410. Split Array Largest Sum -- DP */
/* Given an array which consists of non-negative integers
* and an integer m, you can split the array into m
* non-empty continuous subarrays. Write an algorithm to
* minimize the largest sum among these m subarrays.
* Note: If n is the length of array, assume the following
* constraints are satisfied:
* 1 ≤ n ≤ 1000, 1 ≤ m ≤ min(50, n)
* Examples: Input: nums = [7,2,5,10,8], m = 2. Output: 18. */
int splitArray(vector<int>& nums, int m) {
	int n = nums.size();
	// IMPORTANT. USE "long" to avoid stack over flow
	vector<long> sums(n + 1, 0);
	for (int i = 1; i <= n; ++i) sums[i] = sums[i - 1] + nums[i - 1];
	// dp[i][j] 表示将数组中前j个数字分成i组所能得到的最小的各个子数组中最大值.
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, INT_MAX));
	dp[0][0] = 0;

	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			// 假如中间任意一个位置k，dp[i-1][k] 表示数组中前k个数字分成 i-1 组
			// 所能得到的最小的各个子数组中最大值，而sums[j]-sums[k]就是后面的数字之和，
			// 取二者之间的较大值，然后和 dp[i][j] 原有值进行对比, 更新dp[i][j] 为二者之中的较小值.
			for (int k = i - 1; k < j; ++k) {
				int t = max(dp[i - 1][k], (int)(sums[j] - sums[k]));
				dp[i][j] = min(dp[i][j], t);
			}
		}
	}
	return dp.back().back();
}

/* 96. Unique Binary Search Trees */
/* Given n, how many structurally unique BST's (binary search trees) that store values 1 ... n?
* Example: Input: 3. Output: 5 */
int numTrees(int n) {
	vector<int> dp(n + 1, 0);
	dp[0] = 1, dp[1] = 1;
	for (int i = 2; i <= n; ++i) {
		for (int j = 0; j < i; ++j) {
			dp[i] += dp[j] * dp[i - j - 1];
		}
	}
	return dp.back();
}

/* 95. Unique Binary Search Trees II */
/* Given an integer n, generate all structurally unique BST's (binary search trees) that store values 1 ... n. */
vector<TreeNode*> *generateTrees(int start, int end) {
	vector<TreeNode*> *res = new vector<TreeNode*>();
	if (start > end) res->push_back(NULL);
	else {
		for (int i = start; i <= end; ++i) {
			vector<TreeNode*> *leftSub = generateTrees(start, i - 1);
			vector<TreeNode*> *rightSub = generateTrees(i + 1, end);

			for (int j = 0; j < (*leftSub).size(); ++j) {
				for (int k = 0; k < rightSub->size(); ++k) {
					TreeNode* node = new TreeNode(i);
					node->left = (*leftSub)[j];
					node->right = (*rightSub)[k];
					res->push_back(node);
				}
			}
		}
	}
	return res;
}

vector<TreeNode*> generateTrees(int n) {
	if (n == 0) return {};
	return *generateTrees(1, n); // pass by reference. 
}




/* ======================== Greedy ======================== */
/* 759. Employee Free Time -- HARD */
/* We are given a list schedule of employees, which represents the working time
* for each employee. Each employee has a list of non-overlapping Intervals,
* and these intervals are in sorted order. Return the list of finite intervals
* representing common, positive-length free time for all employees, also in sorted order.
* Input: schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]. Output: [[5,6],[7,9]].
* Note: schedule and schedule[i] are lists with lengths in range [1, 50].*/
vector<Interval*> employeeFreeTime(vector<vector<Interval*>> schedule) {
	vector<Interval*> res;
	map<int, int> m;
	for (int i = 0; i < schedule.size(); ++i) {
		for (int j = 0; j < schedule[i].size(); ++j) {
			++m[schedule[i][j]->start];
			--m[schedule[i][j]->end];
		}
	}
	int worker = 0, pre = -1;
	for (auto it : m) {
		if (worker == 0 && pre != -1) {
			res.push_back(new Interval(pre, it.first));
		}
		worker += it.second;
		pre = it.first;
	}
	return res;
}



/* ======================== Depth First Search =================== */
/* 110. Balanced Binary Tree */
/* Given a binary tree, determine if it is height-balanced.
* For this problem, a height-balanced binary tree is defined as:
* a binary tree in which the depth of the two subtrees of every node never
* differ by more than 1. */
int height(TreeNode* node) {
	if (!node) return 0;
	return 1 + max(height(node->left), height(node->right));
}

bool isBalanced(TreeNode* root) {
	if (!root) return true;
	return isBalanced(root->left) && isBalanced(root->right) &&
		abs(height(root->left) - height(root->right)) <= 1;
}

/* 100. Same Tree */
bool isSameTree(TreeNode* p, TreeNode* q) {
	if (!p && !q) return true;
	if (!p || !q || p->val != q->val) return false;
	return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

/* 572. Subtree of Another Tree */
/* Given two non-empty binary trees s and t, check whether tree t has exactly the same 
* structure and node values with a subtree of s. A subtree of s is a tree consists of
* a node in s and all of this node's descendants. 
* The tree s could also be considered as a subtree of itself. */
bool isSame(TreeNode* p, TreeNode* q) {
	if (!p && !q) return true;
	if (!p || !q) return false;
	return p->val == q->val && isSame(p->left, q->left) && isSame(p->right, q->right);
}

bool isSubtree(TreeNode* s, TreeNode* t) {
	if (!s) return false;
	if (isSame(s, t)) return true;
	return isSubtree(s->left, t) || isSubtree(s->right, t);
}

/* 104. Maximum Depth of Binary Tree */
int maxDepth(TreeNode* root) {
	if (!root) return 0;
	return 1 + max(maxDepth(root->left), maxDepth(root->right));
}

/* 129. Sum Root to Leaf Numbers */
/* Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
* An example is the root-to-leaf path 1->2->3 which represents the number 123.
* Find the total sum of all root-to-leaf numbers. */
int sumNumbers(TreeNode* root, int res) {
	if (!root) return 0;
	res = res * 10 + root->val;
	if (!root->left && !root->right) return res;
	return sumNumbers(root->left, res) + sumNumbers(root->right, res);
}

int sumNumbers(TreeNode* root) {
	return sumNumbers(root, 0);
}

/* 101. Symmetric Tree */
bool isSymmetric(TreeNode* p, TreeNode* q) {
	if (!p && !q) return true;
	if (!p || !q || (p->val != q->val)) return false;
	return isSymmetric(p->left, q->right) && isSymmetric(p->right, q->left);
}

bool isSymmetric(TreeNode* root) {
	if (!root) return true;
	return isSymmetric(root, root);
}

/* 226. Invert Binary Tree */
TreeNode* invertTree(TreeNode* root) {
	if (!root) return NULL;
	TreeNode* t = root->left;
	root->left = invertTree(root->right);
	root->right = invertTree(t);
	return root;
}

/* 543. Diameter of Binary Tree */
/* Given a binary tree, you need to compute the length of the diameter of the tree.
* The diameter of a binary tree is the length of the longest path between any
* two nodes in a tree. This path may or may not pass through the root.
* (1) 对每一个结点求出其左右子树深度之和，这个值作为一个候选值.
* (2) 然后再对左右子结点分别调用求直径对递归函数，这三个值相互比较，取最大的值更新结果res*/
int getHeight(TreeNode* root) {
	if (!root) return 0;
	return 1 + max(getHeight(root->left), getHeight(root->right));
}

int diameterOfBinaryTree(TreeNode* root) {
	if (!root) return 0;
	int res = 0;
	res = getHeight(root->left) + getHeight(root->right);
	return max(res, max(diameterOfBinaryTree(root->left), diameterOfBinaryTree(root->right)));
}

/* 94. Binary Tree Inorder Traversal */
vector<int> inorderTraversal(TreeNode* root) {
	vector<int> res;
	if (!root) return res;
	stack<TreeNode*> st;
	TreeNode* p = root;
	while (!st.empty() || p) {
		while (p) {
			st.push(p);
			p = p->left;
		}
		p = st.top(); st.pop();
		res.push_back(p->val);
		p = p->right;
	}
	return res;
}

/* 314. Binary Tree Vertical Order Traversal */
vector<vector<int>> verticalOrder(TreeNode* root) {
	vector<vector<int>> res;
	if (!root) return res;
	queue<pair<TreeNode*, int>> q;
	q.push({ root, 0 });
	map<int, vector<int>> m;

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		m[t.second].push_back(t.first->val);
		if (t.first->left) q.push({ t.first->left, t.second - 1 });
		if (t.first->right) q.push({ t.first->right, t.second + 1 });
	}

	for (auto it : m) {
		res.push_back(it.second);
	}

	return res;
}

/* 103. Binary Tree Zigzag Level Order Traversal */
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
	vector<vector<int>> res;
	if (!root) return res;
	stack<TreeNode*> st1, st2;
	st1.push(root);

	while (!st1.empty() || !st2.empty()) {
		vector<int> ind;
		while (!st1.empty()) {
			auto t = st1.top(); st1.pop();
			ind.push_back(t->val);
			if (t->left) st2.push(t->left);
			if (t->right) st2.push(t->right);
		}
		if (!ind.empty()) res.push_back(ind);
		ind.clear();

		while (!st2.empty()) {
			auto t = st2.top(); st2.pop();
			ind.push_back(t->val);

			if (t->right) st1.push(t->right);
			if (t->left) st1.push(t->left);
		}
		if (!ind.empty()) res.push_back(ind);
		ind.clear();
	}

	return res;
}

/* 108. Convert Sorted Array to Binary Search Tree */
/* Given an array where elements are sorted in ascending order,
* convert it to a height balanced BST.*/
TreeNode* sortedArrayToBST(vector<int>& nums, int l, int r) {
	if (l > r) return NULL;
	int ix = l + (r - l) / 2;
	TreeNode* root = new TreeNode(nums[ix]);
	root->left = sortedArrayToBST(nums, l, ix - 1);
	root->right = sortedArrayToBST(nums, ix + 1, r);
	return root;
}

TreeNode* sortedArrayToBST(vector<int>& nums) {
	if (nums.empty()) return NULL;
	return sortedArrayToBST(nums, 0, nums.size() - 1);
}

/* 109. Convert Sorted List to Binary Search Tree. */
/* Given a singly linked list where elements are sorted in
* ascending order, convert it to a height balanced BST.*/
TreeNode* sortedListToBST(ListNode* head) {
	if (!head) return NULL;
	ListNode* slow = head, *fast = head, *pre = head;
	while (fast->next && fast->next->next) {
		pre = slow;
		slow = slow->next;
		fast = fast->next->next;
	}
	TreeNode* root = new TreeNode(slow->val);
	fast = slow->next;
	slow->next = NULL;
	pre->next = NULL;
	if (slow != head) root->left = sortedListToBST(head);
	root->right = sortedListToBST(fast);
	return root;
}

/* 124. Binary Tree Maximum Path Sum */
/* Given a non-empty binary tree, find the maximum path sum. */
int maxPathSum(TreeNode* node, int& res) {
	if (!node) return 0;
	int left = max(maxPathSum(node->left, res), 0);
	int right = max(maxPathSum(node->right, res), 0);
	res = max(res, left + right + node->val);
	return max(left, right) + node->val;
}

int maxPathSum(TreeNode* root) {
	if (!root) return 0;
	int res = INT_MIN;
	maxPathSum(root, res);
	return res;
}

/* 270. Closest Binary Search Tree Value */
/* Given a non-empty binary search tree and a target value, find the value in the BST
* that is closest to the target. Given target value is a floating point. You are guaranteed
* to have only one unique value in the BST that is closest to the target.*/
int closestValue(TreeNode* root, double target) {
	int res = root->val;
	while (root) {
		if (abs(target - root->val) <= abs(target - res)) res = root->val;
		root = root->val > target ? root->left : root->right;
	}
	return res;
}

/* 235. Lowest Common Ancestor of a Binary Search Tree */
/* Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST. */
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
	if (!root || !p || !q) return NULL;
	while (true) {
		if (root->val > max(p->val, q->val)) return lowestCommonAncestor(root->left, p, q);
		else if (root->val < min(p->val, q->val)) return lowestCommonAncestor(root->right, p, q);
		else return root;
	}
}

/* 236. Lowest Common Ancestor of a Binary Tree */
TreeNode* lowestCommonAncestor2(TreeNode* root, TreeNode* p, TreeNode* q) {
	if (!root || p == root || q == root) return root;
	TreeNode* left = lowestCommonAncestor2(root->left, p, q);
	TreeNode* right = lowestCommonAncestor2(root->right, p, q);
	if (left && right) return root;
	else return left ? left : right;
}

// 797. All Paths From Source to Target
void allPathsSourceTarget(vector<vector<int>>& graph, int cur, vector<int>& ind, vector<vector<int>>& res) {
	ind.push_back(cur);
	if (cur == graph.size() - 1) {
		res.push_back(ind);
		return;
	}

	for (auto a : graph[cur]) {
		allPathsSourceTarget(graph, a, ind, res);
	}
}

vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
	vector<vector<int>> res;
	vector<int> ind;
	allPathsSourceTarget(graph, 0, ind, res);
	return res;
}

/* 394. Decode String */
/* s = "3[a]2[bc]", return "aaabcbc".
* s = "3[a2[c]]", return "accaccacc".
* s = "2[abc]3[cd]ef", return "abcabccdcdcdef". */
string decodeString(int& i, string s) {
	string res("");
	int n = s.size();
	// IMPORTANT: Use "&&" here
	while (i < n && s[i] != ']') {
		if (s[i] < '0' || s[i] > '9') {
			res += s[i++];
		}
		else {
			int cnt = 0;
			while (s[i] >= '0' && s[i] <= '9') {
				cnt = cnt * 10 + (s[i++] - '0');
			}
			++i;
			string t = decodeString(i, s);
			++i;
			// IMPORTANT: use postfix here
			while (cnt-- > 0) res += t;
		}
	}
	return res;
}

string decodeString(string s) {
	int i = 0;
	return decodeString(i, s);
}

/* 399. Evaluate Division */
/* Equations are given in the format A / B = k, where A and B are variables
* represented as strings, and k is a real number (floating point number).
* Given some queries, return the answers. If the answer does not exist, return -1.0. */
double calcEquation(string up, string down, unordered_map<string, unordered_map<string, double>>& m,
	unordered_set<string>& visited) {
	if (m[up].count(down)) return m[up][down];
	for (auto a : m[up]) {
		if (visited.find(a.first) == visited.end()) {
			visited.insert(a.first);
			auto t = calcEquation(a.first, down, m, visited);
			if (t) return a.second * t;
		}
	}
	return 0;
}

vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values,
	vector<vector<string>>& queries) {
	vector<double> res;
	unordered_map<string, unordered_map<string, double>> m;
	for (int i = 0; i < equations.size(); ++i) {
		m[equations[i][0]][equations[i][1]] = values[i];
		if (values[i]) m[equations[i][1]][equations[i][0]] = 1.0 / values[i];
	}

	for (auto a : queries) {
		unordered_set<string> visited;
		auto ind = calcEquation(a[0], a[1], m, visited);
		if (ind) res.push_back(ind);
		else res.push_back(-1);
	}
	return res;
}

/* 200. Number of Islands */
void numIslands(vector<vector<char>>& grid, vector<vector<int>>& visited, int i, int j) {
	if (visited[i][j]) return;
	visited[i][j] = 1;
	int m = grid.size(), n = grid[0].size();

	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		if (x >= 0 && x < m && y >= 0 && y < n && !visited[x][y] && grid[x][y] == '1') {
			numIslands(grid, visited, x, y);
		}
	}
}

int numIslands(vector<vector<char>>& grid) {
	if (grid.empty() || grid[0].empty()) return 0;
	int m = grid.size(), n = grid[0].size(), res = 0;
	vector<vector<int>> visited(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (!visited[i][j] && grid[i][j] == '1') {
				++res;
				numIslands(grid, visited, i, j);
			}
		}
	}
	return res;
}

/* Validate Binary Search Tree */
bool isValidBST(TreeNode* node, long mn, long mx) { // USE "LONG" type
	if (!node) return true; // IMPORTANT
	if (node->val <= mn || node->val >= mx) return false; 
	return isValidBST(node->left, mn, node->val) && isValidBST(node->right, node->val, mx);
}
bool isValidBST(TreeNode* root) {
	if (!root) return true;
	return isValidBST(root, LONG_MIN, LONG_MAX);
}

/* Word Search */
/* Given a 2D board and a word, find if the word exists in the grid.
* The word can be constructed from letters of sequentially adjacent cell,
* where "adjacent" cells are those horizontally or vertically neighboring.
* The same letter cell may not be used more than once. */
bool wordSearch(vector<vector<char>>& board, string word, vector<vector<int>>& visited, int idx, int i, int j) {
	if (idx == word.size()) return true;
	int m = board.size(), n = board[0].size();
	if (i < 0 || i >= m || j < 0 || j >= n || visited[i][j] || board[i][j] != word[idx]) return false;
	visited[i][j] = 1; // IMPORTANT
	bool res = wordSearch(board, word, visited, idx + 1, i + 1, j) ||
		wordSearch(board, word, visited, idx + 1, i - 1, j) ||
		wordSearch(board, word, visited, idx + 1, i, j + 1) ||
		wordSearch(board, word, visited, idx + 1, i, j - 1);
	visited[i][j] = 0; // IMPORTANT
	return res;
}

bool wordSearch(vector<vector<char>>& board, string word) {
	if (board.empty() || board[0].empty()) return false;
	int m = board.size(), n = board[0].size();
	vector<vector<int>> visited(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (wordSearch(board, word, visited, 0, i, j)) return true;
		}
	}
	return false;
}

// k-th smallest element in a BST
int kthSmallestBST(TreeNode* root, int k) {
	if (!root) return -1;
	TreeNode* p = root;
	stack<TreeNode*> st;
	int cnt = 0;

	while (p || !st.empty()) {
		while (p) {
			st.push(p);
			p = p->left;
		}

		p = st.top();
		st.pop();
		++cnt;
		if (cnt == k) return p->val;
		p = p->right;
	}
	return -1;
}

/* 106. Construct Binary Tree from Inorder and Postorder Traversal */
/* Given inorder and postorder traversal of a tree, construct the binary tree.
* You may assume that duplicates do not exist in the tree. */
TreeNode* buildTreeInPo(vector<int>& inorder, int ileft, int iright, vector<int>& postorder, int pleft, int pright) {
	if (ileft > iright || pleft > pright) return NULL;
	TreeNode* root = new TreeNode(postorder[pright]);
	// IMPORTANT. Scope of "ix"
	int ix = 0;
	for (ix = 0; ix < inorder.size(); ++ix) {
		if (inorder[ix] == root->val) {
			break;
		}
	}
	root->left = buildTreeInPo(inorder, ileft, ix - 1, postorder, pleft, pleft + ix - ileft - 1);
	root->right = buildTreeInPo(inorder, ix + 1, iright, postorder, pleft + ix - ileft, pright - 1);
	return root;
}

TreeNode* buildTreeInPo(vector<int>& inorder, vector<int>& postorder) {
	if (inorder.size() != postorder.size()) return NULL;
	int n = inorder.size();
	return buildTreeInPo(inorder, 0, n - 1, postorder, 0, n - 1);
}

/* 105. Construct Binary Tree from Preorder and Inorder Traversal */
/* You may assume that duplicates do not exist in the tree. */
TreeNode* buildTreeInPre(vector<int>& preorder, int pleft, int pright, vector<int>& inorder, int ileft, int iright) {
	if (pleft > pright || ileft > iright) return NULL;
	TreeNode* root = new TreeNode(preorder[pleft]);
	int ix = 0;
	for (ix = 0; ix < inorder.size(); ++ix) {
		if (inorder[ix] == root->val) break;
	}
	root->left = buildTreeInPre(preorder, pleft + 1, ix - ileft + pleft, inorder, ileft, ix - 1);
	root->right = buildTreeInPre(preorder, ix - ileft + pleft + 1, pright, inorder, ix + 1, iright);
	return root;
}

TreeNode* buildTreeInPre(vector<int>& preorder, vector<int>& inorder) {
	if (preorder.size() != inorder.size()) return NULL;
	int n = preorder.size();
	return buildTreeInPre(preorder, 0, n - 1, inorder, 0, n - 1);
}

/* 250. Count Univalue Subtrees */
/* Given a binary tree, count the number of uni-value subtrees.
* A Uni-value subtree means all nodes of the subtree have the same value. */
bool isUnival(TreeNode* node, int val) {
	if (!node) return true;
	return node->val == val && isUnival(node->left, node->val) && isUnival(node->right, node->val);
}

int countUnivalSubtrees(TreeNode* root, int& res) {
	if (!root) return 0;
	if (isUnival(root, root->val)) ++res;

	countUnivalSubtrees(root->left, res);
	countUnivalSubtrees(root->right, res);
	return res;
}

int countUnivalSubtrees(TreeNode* root) {
	int res = 0;
	return countUnivalSubtrees(root, res);
}

/* 207. Course Schedule -- 染色大法 DFS */
/* There are a total of n courses you have to take,
* labeled from 0 to n-1. Some courses may have prerequisites,
* for example to take course 0 you have to first take course 1,
* which is expressed as a pair: [0, 1].
* Given the total number of courses and a list of prerequisite
* pairs, is it possible for you to finish all courses?
* Input: 2, [[1,0]]. Output: true. */
bool canFinish(vector<vector<int>>& g, vector<int>& visited, int cur) {
	if (visited[cur] != -1) return visited[cur];
	visited[cur] = 0;

	for (auto a : g[cur]) {
		if (!canFinish(g, visited, a)) return false;
	}

	visited[cur] = 1;
	return true;
}

bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
	vector<int> visited(numCourses, -1);
	vector<vector<int>> g(numCourses, vector<int>());

	for (auto a : prerequisites) {
		g[a[1]].push_back(a[0]);
	}
	for (int i = 0; i < numCourses; ++i) {
		if (!canFinish(g, visited, i)) return false;
	}
	return true;
}

/* 210. Course Schedule II */
/* There are a total of n courses you have to take,
* labeled from 0 to n-1. Some courses may have prerequisites,
* for example to take course 0 you have to first take course 1,
* which is expressed as a pair: [0,1]. Given the total number
* of courses and a list of prerequisite pairs, return the
* ordering of courses you should take to finish all courses.
* There may be multiple correct orders, you just need to
* return one of them. If it is impossible to finish all courses,
* return an empty array.*/
vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
	vector<vector<int>> g(numCourses, vector<int>());
	vector<int> in(numCourses, 0);
	queue<int> q;
	vector<int> res;

	for (auto a : prerequisites) {
		g[a[1]].push_back(a[0]);
		++in[a[0]];
	}

	for (int i = 0; i < numCourses; ++i) {
		if (in[i] == 0) q.push(i);
	}

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		res.push_back(t);
		for (auto a : g[t]) {
			--in[a];
			if (in[a] == 0) q.push(a);
		}
	}
	if (res.size() != numCourses) res.clear();
	return res;
}

// 993. Cousins in Binary Tree
/* In a binary tree, the root node is at depth 0, and
* children of each depth k node are at depth k+1.
* Two nodes of a binary tree are cousins if they have
* the same depth, but have different parents.
* We are given the root of a binary tree with unique values,
* and the values x and y of two different nodes in the tree.
* Return true if and only if the nodes corresponding to
* the values x and y are cousins. */
void isCousins(TreeNode* node, TreeNode* parent, int depth, unordered_map<int, pair<TreeNode*, int>>& m) {
	if (!node) return;
	m[node->val] = { parent, depth };
	isCousins(node->left, node, depth + 1, m);
	isCousins(node->right, node, depth + 1, m);
}

bool isCousins(TreeNode* root, int x, int y) {
	unordered_map<int, pair<TreeNode*, int>> m;
	isCousins(root, NULL, 0, m);
	auto a = m[x], b = m[y];
	return a.second == b.second && a.first != b.first;
}

/* 547. Friend Circles -- UNDIRECTED GRAPH PROBLEM */
/* There are N students in a class. Some of them are friends,
* while some are not. Their friendship is transitive in nature.
* For example, if A is a direct friend of B, and B is a
* direct friend of C, then A is an indirect friend of C.
* And we defined a friend circle is a group of students
* who are direct or indirect friends.
* Given a N*N matrix M representing the friend relationship
* between students in the class. If M[i][j] = 1,
* then the ith and jth students are direct friends with each other,
* otherwise not. And you have to output the total number of
* friend circles among all the students.*/
void findCircleNum(vector<vector<int>>& M, vector<int>& visited, int cur) {
	if (visited[cur]) return;
	visited[cur] = 1;
	for (int i = 0; i < M.size(); ++i) {
		if (!M[cur][i] || visited[i]) continue;
		findCircleNum(M, visited, i);
	}
}

int findCircleNum(vector<vector<int>>& M) {
	int n = M.size(), res = 0;
	vector<int> visited(n, 0);
	for (int i = 0; i < n; ++i) {
		if (visited[i]) continue; // IMPORTANT
		findCircleNum(M, visited, i);
		++res;
	}
	return res;
}

/* 695. Max Area of Island */
/* Given a non-empty 2D array grid of 0's and 1's,
* an island is a group of 1's (representing land) connected
* 4-directionally (horizontal or vertical.) You may assume
* all four edges of the grid are surrounded by water.
* Find the maximum area of an island in the given 2D array.
* (If there is no island, the maximum area is 0.) */
void maxAreaOfIsland(vector<vector<int>>& grid, int i, int j, int& res) {
	if (grid[i][j] == 2) return; // visited check
	grid[i][j] = 2;
	int m = grid.size(), n = grid[0].size();

	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == 1) {
			++res;
			maxAreaOfIsland(grid, x, y, res);
		}
	}
}

int maxAreaOfIsland(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size(), res = 0;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == 1) {
				int t = 1;
				maxAreaOfIsland(grid, i, j, t);
				res = max(res, t);
			}
		}
	}
	return res;
}

/* 529. Minesweeper */
/* You are given a 2D char matrix representing the game board.
* (1) 'M' represents an unrevealed mine,
* (2) 'E' represents an unrevealed empty square,
* (3) 'B' represents a revealed blank square that has no adjacent
*     (above, below, left, right, and all 4 diagonals) mines,
* (4) digit ('1' to '8') represents how many mines are adjacent to
*     this revealed square, and finally 'X' represents a revealed mine.
* Now given the next click position (row and column indices)
* among all the unrevealed squares ('M' or 'E'), return the
* board after revealing this position according to the following rules:
* (1) If a mine ('M') is revealed, then the game is over - change it to 'X'.
* (2) If an empty square ('E') with no adjacent mines is revealed,
*     then change it to revealed blank ('B') and all of its
*     adjacent unrevealed squares should be revealed recursively.
* (3) If an empty square ('E') with at least one adjacent mine is
*     revealed, then change it to a digit ('1' to '8') representing
*     the number of adjacent mines. Return the board when no more
*     squares will be revealed. */
vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
	int m = board.size(), n = board[0].size(), r = click[0], c = click[1];
	if (board[r][c] == 'M') {
		board[r][c] = 'X';
	}
	else {
		int cnt = 0;
		for (int i = -1; i < 2; ++i) {
			for (int j = -1; j < 2; ++j) {
				int x = r + i, y = c + j;
				if (x >= 0 && x < m && y >= 0 && y < n && board[x][y] == 'M') {
					++cnt;
				}
			}
		}
		if (cnt == 0) {
			board[r][c] = 'B';
			for (int i = -1; i < 2; ++i) {
				for (int j = -1; j < 2; ++j) {
					int x = r + i, y = c + j;
					if (x >= 0 && x < m && y >= 0 && y < n && board[x][y] == 'E') {
						vector<int> nextClick{ x, y };
						updateBoard(board, nextClick);
					}
				}
			}
		}
		else {
			board[r][c] = cnt + '0';
		}
	}
	return board;
}

/* 694. Number of Distinct Islands */
/* Given a non-empty 2D array grid of 0's and 1's, an island
* is a group of 1's (representing land) connected
* 4-directionally (horizontal or vertical.) You may
* assume all four edges of the grid are surrounded by water.
* Count the number of distinct islands. An island is
* considered to be the same as another if and only if
* one island can be translated (and not rotated or reflected)
* to equal the other.*/
void numDistinctIslands(vector<vector<int>>& grid, vector<vector<int>>& visited, int x0, int y0, int i, int j, set<string>& st) {
	if (visited[i][j]) return;
	visited[i][j] = 1;
	int m = grid.size(), n = grid[0].size();

	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		if (x >= 0 && x < m && y >= 0 && y < n && !visited[x][y] && grid[x][y] == 1) {
			st.insert(to_string(x - x0) + "_" + to_string(y - y0));
			numDistinctIslands(grid, visited, x0, y0, x, y, st);
		}
	}
}

int numDistinctIslands(vector<vector<int>>& grid) {
	set<string> res;
	int m = grid.size(), n = grid[0].size();
	vector<vector<int>> visited(m, vector<int>(n, 0));
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (!visited[i][j] && grid[i][j] == 1) {
				set<string> st;
				numDistinctIslands(grid, visited, i, j, i, j, st);
				string s("");
				for (auto a : st) s += a;
				res.insert(s);
			}
		}
	}
	return res.size();
}

/* 417. Pacific Atlantic Water Flow */
/* Given an m x n matrix of non-negative integers representing
* the height of each unit cell in a continent, the "Pacific ocean"
* touches the left and top edges of the matrix and the
* "Atlantic ocean" touches the right and bottom edges.
* Water can only flow in four directions (up, down, left, or right)
* from a cell to another one with height equal or lower.
* Find the list of grid coordinates where water can flow to
* both the Pacific and Atlantic ocean. */
void pacificAtlantic(vector<vector<int>>& matrix, vector<vector<int>>& visited, int i, int j, int pre) {
	int m = matrix.size(), n = matrix[0].size();
	// IMPORTANT. TO TEST BOUNDARY HERE AHEAD. 
	if (i < 0 || i >= m || j < 0 || j >= n || visited[i][j] || matrix[i][j] < pre) return;
	visited[i][j] = 1;

	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		pacificAtlantic(matrix, visited, x, y, matrix[i][j]);
	}
}

vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
	vector<vector<int>> res;
	if (matrix.empty() || matrix[0].empty()) return res;
	int m = matrix.size(), n = matrix[0].size();
	vector<vector<int>> pacific(m, vector<int>(n, 0));
	vector<vector<int>> atlantic(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		pacificAtlantic(matrix, pacific, i, 0, INT_MIN);
		pacificAtlantic(matrix, atlantic, i, n - 1, INT_MIN);
	}

	for (int j = 0; j < n; ++j) {
		pacificAtlantic(matrix, pacific, 0, j, INT_MIN);
		pacificAtlantic(matrix, atlantic, m - 1, j, INT_MIN);
	}

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (atlantic[i][j] && pacific[i][j]) {
				res.push_back({ i, j });
			}
		}
	}
	return res;
}

/* 112. Path Sum*/
/* Given a binary tree and a sum, determine if the tree
* has a root-to-leaf path such that adding up all the
* values along the path equals the given sum. */
bool hasPathSum(TreeNode* root, int sum) {
	if (!root) return false;
	// IMPORTANT. "sum == root -> val"
	if (!root->left && !root->right && sum == root->val) return true;
	else return hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val);
}

/* 113. Path Sum II*/
/* Given a binary tree and a sum, find all root-to-leaf
* paths where each path's sum equals the given sum. */
void pathSum(TreeNode* node, int sum, vector<int>& ind, vector<vector<int>>& res) {
	ind.push_back(node->val);
	if (!node->left && !node->right && node->val == sum) {
		res.push_back(ind);
		return;
	}
	if (node->left) pathSum(node->left, sum - node->val, ind, res);
	if (node->right) pathSum(node->right, sum - node->val, ind, res);
	ind.pop_back(); // IMPORTANT.
}

vector<vector<int>> pathSum(TreeNode* root, int sum) {
	vector<vector<int>> res;
	vector<int> ind;
	pathSum(root, sum, ind, res);
	return res;
}

/* 332. Reconstruct Itinerary */
/* Given a list of airline tickets represented by pairs of departure and arrival airports [from, to],
* reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK.
* Thus, the itinerary must begin with JFK.
* Input: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
* Output: ["JFK", "MUC", "LHR", "SFO", "SJC"]. */
void findItinerary(string cur, unordered_map<string, multiset<string>>& m, vector<string>& res) {
	while (m[cur].size() > 0) {
		auto a = *m[cur].begin();
		m[cur].erase(m[cur].begin());
		findItinerary(a, m, res);
	}
	res.push_back(cur);
}

vector<string> findItinerary(vector<vector<string>>& tickets) {
	vector<string> res;
	unordered_map<string, multiset<string>> m;
	for (auto a : tickets) m[a[0]].insert(a[1]);
	findItinerary("JFK", m, res);
	return vector<string>(res.rbegin(), res.rend());
}

/* 54. Spiral Matrix */
/* Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order. */
vector<int> spiralOrder(vector<vector<int>>& matrix) {
	if (matrix.empty() || matrix[0].empty()) return {};
	vector<int> res;
	vector<vector<int>> dirs{ { 0, 1 },{ 1, 0 },{ 0, -1 },{ -1, 0 } };
	int m = matrix.size(), n = matrix[0].size(), idx = 0, i = 0, j = 0;

	for (int k = 0; k < m * n; ++k) {
		res.push_back(matrix[i][j]);
		matrix[i][j] = 0;

		int x = i + dirs[idx][0], y = j + dirs[idx][1];
		if (x < 0 || x >= m || y < 0 || y >= n || matrix[x][y] == 0) {
			idx = (idx + 1) % 4;
			x = i + dirs[idx][0];
			y = j + dirs[idx][1];
		}
		i = x;
		j = y;
	}
	return res;
}

/* 59. Spiral Matrix II */
/* Given a positive integer n, generate a square matrix filled with elements from 1 to n^2 in spiral order. */
/*
vector<vector<int>> generateMatrix(int n) {
} */


/* ======================== Breath First Search ===================== */
/* 102. Binary Tree Level Order Traversal */
vector<vector<int>> levelOrder(TreeNode* root) {
	vector<vector<int>> res;
	if (!root) return res;
	queue<TreeNode*> q{ { root } };
	while (!q.empty()) {
		int n = q.size();
		vector<int> ind;
		for (int i = 0; i < n; ++i) {
			auto t = q.front(); q.pop();
			ind.push_back(t->val);
			if (t->left) q.push(t->left);
			if (t->right) q.push(t->right);
		}
		res.push_back(ind);
	}
	return res;
}

/* 199. Binary Tree Right Side View */
/* Given a binary tree, imagine yourself standing on the right side of it,
* return the values of the nodes you can see ordered from top to bottom. */
vector<int> rightSideView(TreeNode* root) {
	vector<int> res;
	if (!root) return res;
	queue<TreeNode*> q{ { root } };
	while (!q.empty()) {
		int n = q.size();
		for (int i = 0; i < n; ++i) {
			auto t = q.front(); q.pop();
			if (i == n - 1) res.push_back(t->val);
			if (t->left) q.push(t->left);
			if (t->right) q.push(t->right);
		}
	}
	return res;
}

/* Remove invalid parenthesis */
/* Remove the minimum number of invalid parentheses in order to make the input string valid.
* Return all possible results. Input: "()())()". Output: ["()()()", "(())()"] */
bool isValidParenthesis(string s) {
	int cnt = 0;
	for (auto c : s) {
		if (c == '(') ++cnt;
		else if (c == ')' && --cnt < 0) return false;
	}
	return cnt == 0;
}

vector<string> removeInvalidParentheses(string s) {
	vector<string> res;
	unordered_set<string> visited{ { s } };
	queue<string> q{ { s } };
	bool found = false;

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		if (isValidParenthesis(t)) {
			res.push_back(t);
			found = true;
		}
		if (found) continue;

		for (int i = 0; i < t.size(); ++i) {
			if (t[i] != '(' && t[i] != ')') continue;
			string temp = t.substr(0, i) + t.substr(i + 1);
			if (!visited.count(temp)) {
				q.push(temp);
				visited.insert(temp);
			}
		}
	}
	return res;
}

/* 127. Word Ladder */
/* Given two words (beginWord and endWord), and a dictionary's word list,
* find the length of SHORTEST transformation sequence from beginWord to endWord,
* such that:
* (1) Only one letter can be changed at a time.
* (2) Each transformed word must exist in the word list.
* Note that beginWord is not a transformed word.
* Input:beginWord = "hit", endWord = "cog",
* wordList = ["hot","dot","dog","lot","log","cog"]. Output: 5. */
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
	unordered_set<string> dicts(wordList.begin(), wordList.end());
	unordered_map<string, int> m{ { beginWord, 1 } };
	queue<string> q{ { beginWord } };

	while (!q.empty()) {
		auto str = q.front(); q.pop();
		for (int i = 0; i < str.size(); ++i) {
			string t = str;
			for (auto c = 'a'; c <= 'z'; ++c) {
				t[i] = c;
				if (dicts.count(t)) {
					if (t == endWord) return m[str] + 1;
					else if (!m.count(t)) {
						m[t] = m[str] + 1;
						q.push(t);
					}
				}
			}
		}
	}
	return 0;
}

/* 126. Word Ladder II */
/* Find all shortest transformation sequence(s) from beginWord to endWord.
* Input: beginWord = "hit", endWord = "cog",
* wordList = ["hot","dot","dog","lot","log","cog"]
* Output:[["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]. */
vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
	vector<vector<string>> res;
	unordered_set<string> dicts(wordList.begin(), wordList.end());

	vector<string> path{ beginWord };
	queue<vector<string>> q{ { path } };
	unordered_set<string> visited;

	int level = 1, minLevel = INT_MAX;

	while (!q.empty()) {
		auto t = q.front(); q.pop();

		if (t.size() > level) {
			for (auto s : visited) dicts.erase(s);
			visited.clear();
			level = t.size();
			if (level > minLevel) break;
		}

		auto last = t.back();
		for (int i = 0; i < last.size(); ++i) {
			string newLast = last;
			for (auto c = 'a'; c <= 'z'; ++c) {
				newLast[i] = c;

				if (!dicts.count(newLast)) continue;
				visited.insert(newLast);

				vector<string> newT = t;
				newT.push_back(newLast);

				if (newLast == endWord) {
					res.push_back(newT);
					minLevel = level;
				}
				else {
					q.push(newT);
				}
			}
		}
	}
	return res;
}

/* 721. Accounts Merge */
/* Given a list accounts, each element accounts[i] is a list of strings,
* where the first element accounts[i][0] is a name,
* and the rest of the elements are emails representing emails of the account.
* Now, we would like to merge these accounts.
* Two accounts definitely belong to the same person if there is
* some email that is common to both accounts.
* Input: accounts =
* [["John", "johnsmith@mail.com", "john00@mail.com"],
* ["John", "johnnybravo@mail.com"],
* ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
* ["Mary", "mary@mail.com"]]
* Output:
* [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],
* ["John", "johnnybravo@mail.com"],
* ["Mary", "mary@mail.com"]]. */
vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
	vector<vector<string>> res;
	int n = accounts.size();
	// can use "set" but "vector" is faster
	unordered_map<string, vector<int>> m;
	vector<int> visited(n, 0);

	for (int i = 0; i < n; ++i) {
		for (int j = 1; j < accounts[i].size(); ++j) {
			// IMPORTANT!!! SAVE THE INDEX
			m[accounts[i][j]].push_back(i);
		}
	}

	for (int i = 0; i < n; ++i) {
		if (visited[i]) continue;
		visited[i] = 1;
		queue<int> q;
		q.push(i);
		set<string> st; // SCOPE!!!

		while (!q.empty()) {
			auto t = q.front(); q.pop();
			vector<string> mails(accounts[t].begin() + 1, accounts[t].end());

			for (auto mail : mails) {
				st.insert(mail);
				for (auto a : m[mail]) {
					if (visited[a]) continue;
					visited[a] = 1; // IMPORTANT
					q.push(a);
				}
			}
		}
		vector<string> ind(st.begin(), st.end());
		ind.insert(ind.begin(), accounts[i][0]);
		res.push_back(ind);
	}
	return res;
}

/* 863. All Nodes Distance K in Binary Tree */
/* We are given a binary tree (with root node root), a target node, and an integer value K.
* Return a list of the values of all nodes that have a distance K from the target node.
* The answer can be returned in any order. */
void buildTreeMap(TreeNode* node, TreeNode* pre, unordered_map<TreeNode*, vector<TreeNode*>>& m) {
	if (!node) return;
	if (m.count(node)) return;
	if (pre) {
		m[node].push_back(pre);
		m[pre].push_back(node);
	}
	buildTreeMap(node->left, node, m);
	buildTreeMap(node->right, node, m);
}

vector<int> distanceK(TreeNode* root, TreeNode* target, int K) {
	vector<int> res;
	unordered_map<TreeNode*, vector<TreeNode*> > m;
	buildTreeMap(root, NULL, m);
	unordered_set<TreeNode*> visited{ { target } };
	queue<TreeNode*> q{ { target } };

	while (!q.empty()) {
		if (K == 0) {
			for (int i = q.size() - 1; i >= 0; --i) {
				auto t = q.front(); q.pop();
				res.push_back(t->val);
			}
		}

		for (int i = q.size() - 1; i >= 0; --i) {
			auto t = q.front(); q.pop();
			for (auto a : m[t]) {
				if (!visited.count(a)) {
					visited.insert(a);
					q.push(a);
				}
			}
		}
		--K;
	}
	return res;
}

// 787. Cheapest Flights Within K Stops
/* There are n cities connected by m flights. Each fight starts from
* city u and arrives at v with a price w. Now given all the cities
* and flights, together with starting city src and the destination dst,
* your task is to find the cheapest price from src to dst with up to
* k stops. If there is no such route, output -1. */
int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) {
	unordered_map<int, unordered_map<int, int> > m;
	for (auto a : flights) {
		m[a[0]][a[1]] = a[2];
	}
	queue<pair<int, int>> q;
	q.push({ src, 0 }); // IMPORTANT.
	int res = INT_MAX, k = 0;

	while (!q.empty()) {
		int n = q.size();
		for (int i = 0; i < n; ++i) {
			auto t = q.front(); q.pop();
			if (t.first == dst) res = min(res, t.second);

			for (auto a : m[t.first]) {
				if (t.second + a.second > res) continue;
				q.push({ a.first, t.second + a.second });
			}
		}
		if (k++ > K) break;
	}
	return res == INT_MAX ? -1 : res;
}

/* 733. Flood Fill */
/* An image is represented by a 2-D array of integers,
* each integer representing the pixel value of the image
* (from 0 to 65535). Given a coordinate (sr, sc) representing
* the starting pixel (row and column) of the flood fill,
* and a pixel value newColor, "flood fill" the image.
* Replace the color of all of the aforementioned pixels
* with the newColor. */
vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
	if (image.empty() || image[0].empty() || image[sr][sc] == newColor) return image;
	int m = image.size(), n = image[0].size();
	queue<pair<int, int>> q;
	q.push({ sr, sc });
	int val = image[sr][sc];

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		image[t.first][t.second] = newColor;
		for (auto dir : dirs) {
			int x = t.first + dir[0], y = t.second + dir[1];
			if (x >= 0 && x < m && y >= 0 && y < n && image[x][y] == val) q.push({ x, y });
		}
	}
	return image;
}

/* 662. Maximum Width of Binary Tree */
/* Given a binary tree, write a function to get the maximum width of the given tree.
* The width of a tree is the maximum width among all levels. The binary tree has the same
* structure as a full binary tree, but some nodes are null.
* 如果根结点是深度1，那么每一层的结点数就是 2^(n-1)，那么每个结点的位置就是 [1, 2^(n-1)] 中的一个，
* 假设某个结点的位置是i，那么其左右子结点的位置可以直接算出来，为 2*i 和 2*i + 1. */
int widthOfBinaryTree(TreeNode* root) {
	int res = 0;
	queue<pair<TreeNode*, int> > q;
	q.push({ root, 1 });

	while (!q.empty()) {
		int n = q.size(), left = INT_MAX, right = INT_MIN;
		for (int i = 0; i < n; ++i) {
			auto t = q.front().first;
			int val = q.front().second; q.pop();

			left = min(left, val);
			right = max(right, val);

			if (t->left) q.push({ t->left, val << 1 });
			if (t->right) q.push({ t->right, (val << 1) + 1 });

		}
		res = max(res, right - left + 1);
	}
	return res;
}

/* 116. Populating Next Right Pointers in Each Node */
/* Here is perfect binary tree.
* 117. Populating Next Right Pointers in Each Node II
* Here is any binary tree. */
TreeNodeNext* connect(TreeNodeNext* root) {
	if (!root) return NULL;
	queue<TreeNodeNext*> q{ { root } };
	while (!q.empty()) {
		int n = q.size();
		for (int i = 0; i < n; ++i) {
			auto t = q.front(); q.pop();
			// KEY POINT.
			if (i < n - 1) t->next = q.front();
			if (t->left) q.push(t->left);
			if (t->right) q.push(t->right);
		}
	}
	return root;
}

/* 301. Remove Invalid Parentheses */
/* Remove the minimum number of invalid parentheses in order to make
* the input string valid. Return all possible results.
* Note: The input string may contain letters other than the parentheses ( and ).
* Example 1: Input: "()())()". Output: ["()()()", "(())()"].
* Example 2: Input: "(a)())()". Output: ["(a)()()", "(a())()"]. */
bool isValidStr(string s) {
	int cnt = 0;
	for (auto c : s) {
		if (c >= 'a' && c <= 'z') continue;
		else if (c == '(') ++cnt;
		else if (c == ')') {
			if (--cnt < 0) return false;
		}
	}
	return cnt == 0;
}
vector<string> removeInvalidParentheses(string s) {
	vector<string> res;
	queue<string> q{ { s } };
	unordered_set<string> visited{ { s } };
	bool b = false;
	while (!q.empty()) {
		auto t = q.front(); q.pop();
		if (isValidStr(t)) {
			res.push_back(t);
			b = true;
		}
		if (b) continue;
		//string s = t; 
		for (int i = 0; i < t.size(); ++i) {
			if (t[i] >= 'a' && t[i] <= 'z') continue;
			string s = t.substr(0, i) + t.substr(i + 1);
			if (!visited.count(s)) {
				q.push(s);
				visited.insert(s);
			}
		}
	}
	return res;
}


// ==================== TOPO SORT PROBLEMS ========================
/* Topological sorting for Directed Acyclic Graph (DAG) is a linear ordering of vertices.
* Topological Sorting for a graph is not possible if the graph is not a DAG.
* DFS way: time complexity is the same as DFS which is O(V+E). Space is O(V).
* In computer science, applications of this type arise in instruction scheduling,
* ordering of formula cell evaluation when recomputing formula values in spreadsheets,
* logic synthesis, determining the order of compilation tasks to perform in makefiles,
* data serialization, and resolving symbol dependencies in linkers. */

/* 269. Alien Dictionary -- HARD */
/* There is a new alien language which uses the latin alphabet. However, the order
* among letters are unknown to you. You receive a list of non-empty words from the
* dictionary, where words are sorted lexicographically by the rules of this new
* language. Derive the order of letters in this language.
* Input: ["wrt", "wrf", "er", "ett", "rftt"]. Output: "wertf".
* USE BFS topological sort. (1) A set of char pair to save all pairs by comparing with
* every two strings. (2) A set of char to save all unique chars. (3) A vector to save
* the in-degree for each char. (4) A queue to save in-degree 0 chars and do BFS. */
string alienOrder(vector<string>& words) {
	set<pair<char, char>> st;
	unordered_set<char> cha;
	queue<char> q;
	vector<int> in(256);
	string res("");
	int n = words.size();

	for (auto s : words) cha.insert(s.begin(), s.end());
	// 每两个相邻的单词比较，找出顺序 pair，然后我们根据这些 pair 来赋度.
	for (int i = 0; i < n - 1; ++i) {
		int len = min(words[i].size(), words[i + 1].size());
		int j = 0;
		for (j = 0; j < len; ++j) {
			if (words[i][j] != words[i + 1][j]) {
				st.insert({ words[i][j], words[i + 1][j] });
				break; // VERY IMPORTANT. 
			}
		}
		// Boundary condition.
		if (j == len && words[i].size() > words[i + 1].size()) return "";
	}

	for (auto a : st) ++in[a.second];
	for (auto a : cha) {
		if (in[a] == 0) {
			q.push(a);
			res += a;
		}
	}

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		for (auto a : st) {
			if (a.first == t) {
				--in[a.second];
				if (in[a.second] == 0) {
					q.push(a.second);
					res += a.second;
				}
			}
		}
	}
	return res.size() == cha.size() ? res : "";
}


/* ======================== Binary Search ======================== */
/* 162. Find Peak Element （第五类）*/
/* A peak element is an element that is greater than its neighbors.
* Given an input array nums, where nums[i] ≠ nums[i+1],
* find a peak element and return its INDEX. The array may contain
* multiple peaks, in that case return the index to any one of
* the peaks is fine. Input: nums = [1,2,3,1]. Output: 2. */
int findPeakElement(vector<int>& nums) {
	int left = 0, right = nums.size() - 1;
	while (left + 1 < right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] < nums[mid + 1]) left = mid;
		else right = mid;
	}
	if (nums[left] > nums[right]) return left;
	else return right;
}

/*33.  Search in Rotated Sorted Array */
/* Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
* You are given a target value to search. If found in the array return its index, otherwise return -1.
* You may assume no duplicate exists in the array.
* Your algorithm's runtime complexity must be in the order of O(log n).
* Input: nums = [4,5,6,7,0,1,2], target = 0. Output: 4  */
int searchRotatedArray(vector<int>& nums, int target) {
	int n = nums.size(), left = 0, right = n - 1;
	while (left <= right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] == target) return mid;
		else if (nums[mid] <= nums[right]) { // search right side
			if (target >= nums[mid] && target <= nums[right]) {
				left = mid + 1;
			}
			else {
				right = mid - 1;
			}
		}
		else if (nums[mid] > nums[right]) { // search left side
			if (target >= nums[left] && target <= nums[mid]) {
				right = mid - 1;
			}
			else {
				left = mid + 1;
			}
		}
	}
	return -1;
}

/* 81. Search in Rotated Sorted Array II */
/* Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
* (i.e., [0,0,1,2,2,5,6] might become [2,5,6,0,0,1,2]). You are given a target value to search. 
* If found in the array return true, otherwise return false. 
* Input: nums = [2,5,6,0,0,1,2], target = 0. Output: true. */
bool search(vector<int>& nums, int target) {
	int left = 0, right = nums.size() - 1;
	while (left <= right) {
		int mid = left + (right - left) / 2;

		if (nums[mid] == target) return true;
		else if (nums[mid] > nums[left]) {
			if (target <= nums[mid] && target >= nums[left]) right = mid - 1;
			else left = mid + 1;
		}
		else if (nums[mid] < nums[left]) {
			if (target >= nums[mid] && target <= nums[right]) left = mid + 1;
			else right = mid - 1;
		}
		else {
			++left;
		}
	}
	return false;
}

/* 69. Sqrt(x) （第一类拓展）*/
/* Input: 8. Output: 2. */
int mySqrt(int x) {
	if (x <= 1) return x;
	int left = 0, right = x / 2 + 1;

	while (left + 1 < right) {
		long mid = left + (right - left) / 2;
		if (mid * mid == x) return mid;
		else if (mid * mid > x) right = mid;
		else left = mid;
	}
	return left;
}

/* 42. Trapping Rain Water */
/* Given n non-negative integers representing an elevation map where the width of each bar is 1,
* compute how much water it is able to trap after raining. Input: [0,1,0,2,1,0,1,3,2,1,2,1]. Output: 6 */
int trap(vector<int>& height) {
	int n = height.size(), left = 0, right = n - 1, res = 0;
	while (left < right) {
		int mn = min(height[left], height[right]);
		if (height[left] == mn) {
			++left;
			while (left < n && height[left] < mn) res += mn - height[left++];
		}
		else {
			--right;
			while (right >= 0 && height[right] < mn) res += mn - height[right--];
		}
	}
	return res;
}

/* Three sum */
/* Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0?
* Find all unique triplets in the array which gives the sum of zero. */
vector<vector<int>> threeSum(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	int n = nums.size();
	vector<vector<int>> res;
	for (int i = 0; i < n - 2; ++i) {
		if (nums[i] > 0) break; // IMPORTANT
		if (i > 0 && nums[i] == nums[i - 1]) continue; // IMPORTANT

		int target = 0 - nums[i], l = i + 1, r = n - 1;
		while (l < r) {
			int t = nums[l] + nums[r];
			if (t == target) {
				res.push_back({ nums[i], nums[l], nums[r] });
				while (l < r && nums[l] == nums[l + 1]) ++l; // IMPORTANT
				while (l < r && nums[r] == nums[r - 1]) --r; // IMPORTANT
				++l, --r;
			}
			else if (t > target) {
				--r;
			}
			else {
				++l;
			}
		}
	}
	return res;
}

/* Three sum closes */
/* Given an array nums of n integers and an integer target,
* find three integers in nums such that the sum is closest to target.
* Return the sum of the three integers.
* You may assume that each input would have exactly one solution. */
int threeSumClosest(vector<int>& nums, int target) {
	sort(nums.begin(), nums.end());
	int n = nums.size(), res = nums[0] + nums[1] + nums[2];

	for (int i = 0; i < n - 1; ++i) {
		int diff = abs(target - res), l = i + 1, r = n - 1;

		while (l < r) {
			int newsum = nums[i] + nums[l] + nums[r];
			int newdiff = abs(target - newsum);

			if (newsum < target) ++l;
			else --r;

			if (newdiff < diff) {
				diff = newdiff;
				res = newsum;
			}
		}
	}
	return res;
}

/* Valid Triangle Number ??? */
/* Given an array consists of non-negative integers, your task is to count the number of triplets chosen from
* the array that can make triangles if we take them as side lengths of a triangle.*/
int triangleNumber(vector<int>& nums) {
	if (nums.size() < 3) return 0;
	// 三个数字中如果较小的两个数字之和大于第三个数字，那么任意两个数字之和都大于第三个数字
	sort(nums.begin(), nums.end());
	int n = nums.size(), res = 0;

	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			int sum = nums[i] + nums[j], left = j + 1, right = n; // IMPORTANT: "right = n" 
																  // 将这两个数之和sum作为目标值，然后用二分查找法来快速确定第一个小于目标值的数
			while (left < right) {
				int mid = left + (right - left) / 2;
				if (nums[mid] < sum) left = mid + 1;
				else right = mid;
			}
			// 找到这个临界值，那么这之前一直到j的位置之间的数都满足题意，直接加起来即可
			res += right - 1 - j;
		}
	}
	return res;
}

/* 222. Count Complete Tree Nodes */
/* Given a complete binary tree, count the number of nodes.
* CONCEPT:
* "COMPLETE BINARY TREE": A binary tree in which every level,
* except possibly the last, is completely filled, and
* all nodes are as far left as possible. */
int countNodes(TreeNode* root) {
	if (!root) return 0;
	TreeNode* p1 = root, *p2 = root;
	int hleft = 0, hright = 0;

	while (p1) { ++hleft; p1 = p1->left; }
	while (p2) { ++hright; p2 = p2->right; }

	if (hleft == hright) return pow(2.0, hleft) - 1;
	else return 1 + countNodes(root->left) + countNodes(root->right);
}

/* 29. Divide Two Integers
* Given two integers dividend and divisor, divide two integers without
* using multiplication, division and mod operator. Return the quotient
* after dividing dividend by divisor. The integer division should
* truncate toward zero.*/
int divide(int dividend, int divisor) {
	// IMPORTANT BOUNDARY CONDITION. 
	if (divisor == 0 || (dividend == INT_MIN && divisor == -1)) return INT_MAX;
	// IMPORTANT TO USE "LONG"
	long up = abs((long)dividend), down = abs((long)divisor);
	int sign = ((dividend > 0) ^ (divisor > 0)) ? -1 : 1;
	if (down == 1) return sign == 1 ? up : -up;

	int res = 0;
	while (up >= down) {
		long t = down, p = 1;
		while (up >= (t << 1)) {
			t <<= 1;
			p <<= 1;
		}
		res += p;
		up -= t;
	}
	return sign == -1 ? -res : res;
}

/* 34. Find First and Last Position of Element in Sorted Array (第一类拓展）*/
/* Input: nums = [5,7,7,8,8,10], target = 8. Output: [3,4].
* "findSingle" is the same as "704. Binary Search". */
int findSingle(vector<int>& nums, int target) {
	int n = nums.size(), left = 0, right = n - 1;

	while (left <= right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] == target) return mid;
		else if (nums[mid] < target) left = mid + 1;
		else right = mid - 1;
	}
	return -1;
}

vector<int> searchRange(vector<int>& nums, int target) {
	int ix = findSingle(nums, target);
	if (ix == -1) return { -1, -1 };

	int left = ix, right = ix, n = nums.size();
	while (left > 0 && nums[left] == nums[left - 1]) --left;
	while (right < n - 1 && nums[right] == nums[right + 1]) ++right;
	return { left, right };
}

/* 658. Find K Closest Elements */
/* Given a sorted array, two integers k and x, find the k closest elements
* to x in the array. The result should also be sorted in ascending order.
* If there is a tie, the smaller elements are always preferred.
* Input: [1,2,3,4,5], k=4, x=3. Output: [1,2,3,4]. */
vector<int> findClosestElements(vector<int>& arr, int k, int x) {
	while (arr.size() > k) {
		// IMPORTANT. "="
		if (abs(arr[0] - x) <= abs(arr.back() - x)) arr.pop_back();
		else arr.erase(arr.begin());
	}
	return arr;
}

/* 153. Find Minimum in Rotated Sorted Array  （第二类拓展） */
/* Suppose an array sorted in ascending order is rotated at
* some pivot unknown to you beforehand.
* (i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).
* Find the minimum element.
* You may assume NO DUPLICATE exists in the array.
* LOGIC: NO SPECIFIC TARGET TO FIND. USE below template. */
int findMin(vector<int>& nums) {
	if (nums[0] < nums.back()) return nums[0];
	int n = nums.size(), left = 0, right = n - 1;
	while (left + 1 < right) {
		int mid = left + (right - left) / 2;
		// IMPORTANT. THINK TWICE HERE. 
		if (nums[mid] > nums[left]) {
			left = mid;
		}
		else {
			right = mid;
		}
	}
	if (nums[left] < nums[right]) return nums[left];
	else return nums[right];
}

/* 154. Find Minimum in Rotated Sorted Array II （第二类拓展）*/
/* The array MAY CONTAIN duplicates. */
int findMin(vector<int>& nums) {
	int res = INT_MAX, left = 0, right = nums.size() - 1;
	while (left + 1 < right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] == nums[left]) {
			++left;
		}
		else if (nums[mid] > nums[left]) {
			res = min(res, nums[left]);
			left = mid;
		}
		else {
			res = min(res, nums[mid]);
			right = mid;
		}
	}

	if (nums[left] < nums[right]) res = min(res, nums[left]);
	else res = min(res, nums[right]);

	return res;
}

/* 287. Find the Duplicate Number （第四类）*/
/* Given an array nums containing n + 1 integers where each integer
* is between 1 and n (inclusive), prove that at least one duplicate
* number must exist. Assume that there is only one duplicate number,
* find the duplicate one. Input: [1,3,4,2,2]. Output: 2. */
int findDuplicate(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	int n = nums.size(), left = 0, right = n - 1;

	while (left < right) {
		int mid = left + (right - left) / 2;
		int cnt = 0;
		for (int i = 0; i < n; ++i) {
			if (nums[i] <= mid) ++cnt;
		}
		if (cnt <= mid) left = mid + 1;
		else right = mid;
	}
	return nums[left];
}

/* 1060. Missing Element in Sorted Array */
/* Given a sorted array A of unique numbers, find the K-th missing number starting from the
* leftmost number of the array. Example 1: Input: A = [4,7,9,10], K = 1. Output: 5
* Explanation: The first missing number is 5. */
int missingElement(vector<int>& nums, int k) {
	int l = 0, h = nums.size();
	while (l < h) {
		int m = l + (h - l) / 2;
		if (nums[m] - m - k >= nums[0]) {
			h = m;
		}
		else {
			l = m + 1;
		}
	}
	return nums[0] + l + k - 1;
}

/* 268. Missing Number */
/* Given an array containing n distinct numbers taken from 0, 1, 2, ..., n,
* find the one that is missing from the array. Input: [9,6,4,2,3,5,7,0,1]. Output: 8 */
int missingNumber(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	int left = 0, right = nums.size();

	while (left < right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] > mid) right = mid;
		else left = mid + 1;
	}
	return right;
}

/* 852. Peak Index in a Mountain Array （第五类）*/
/* Let's call an array A a mountain if the following properties hold:
* A.length >= 3. There exists some 0 < i < A.length - 1 such that
* A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length - 1].
* Input: [0,2,1,0]. Output: 1. */
int peakIndexInMountainArray(vector<int>& A) {
	int n = A.size(), left = 0, right = n - 1;
	while (left + 1 < right) {
		int mid = left + (right - left) / 2;
		if (A[mid] < A[mid + 1]) left = mid;
		else right = mid;
	}
	if (A[left] > A[right]) return left;
	else return right;
}

/* 50. Pow(x, n) */
/* Implement pow(x, n), which calculates x raised to the power n (xn). */
double myPowHelper(double x, long n) {
	if (n == 0) return 1;
	if (n == 1) return x;
	// IMPORTANT. "if (n < 0)"
	if (n < 0) return 1 / myPowHelper(x, -n);

	auto half = myPowHelper(x, n / 2);
	if (n % 2 == 0) return half * half;
	else return x * half * half;
}

double myPow(double x, int n) {
	return myPowHelper(x, n);
}

/* 240. Search a 2D Matrix II */
/* Write an efficient algorithm that searches for a value in an m x n matrix. 
* This matrix has the following properties:
* Integers in each row are sorted in ascending from left to right.
* Integers in each column are sorted in ascending from top to bottom. */
bool searchMatrix(vector<vector<int>>& matrix, int target) {
	if (matrix.empty() || matrix[0].empty()) return false;
	int x = matrix.size() - 1, y = 0;
	while (x >= 0 && y < matrix[0].size()) {
		if (matrix[x][y] == target) return true;
		else if (matrix[x][y] > target)--x;
		else ++y;
	}
	return false;
}

// WITH API PROBLEMS.
/* 278. First Bad Version
int firstBadVersion(int n) {
int left = 0, right = n - 1;
while (left <= right) {
int mid = left + (right - left) / 2;
if (isBadVersion(mid)) right = mid - 1;
else left = mid + 1;
}
return left;
}  */



/* ======================== Backtracking ========================== */
/* 39. Combination sum */
/* Given a set of candidate numbers (candidates) (without duplicates) and a target number (target),
* find all unique combinations in candidates where the candidate numbers sums to target.
* The same repeated number may be chosen from candidates unlimited number of times. */
void combinationSum(vector<int>& nums, int target, int idx, vector<int>& ind, vector<vector<int>>& res) {
	if (target < 0) return; // IMPORTANT
	if (target == 0 && !ind.empty()) {
		res.push_back(ind);
		return;
	}
	for (int i = idx; i < nums.size(); ++i) {
		ind.push_back(nums[i]);
		combinationSum(nums, target - nums[i], i, ind, res);
		ind.pop_back(); // BACKTRACKING
	}
}

vector<vector<int>> combinationSum(vector<int>& candaiates, int target) {
	vector<vector<int>>  res;
	vector<int> ind;
	sort(candaiates.begin(), candaiates.end());
	combinationSum(candaiates, target, 0, ind, res);
	return res;
}

/* 17. Letter Combinations of a Phone Number */
/* Input: "23". Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]. */
void letterCombinations(string digits, vector<string>& dicts, int ix, string ind, vector<string>& res) {
	if (ix == digits.size())
	{
		res.push_back(ind);
		return;
	}
	string s = dicts[digits[ix] - '2'];
	for (int i = 0; i < s.size(); ++i) {
		ind.push_back(s[i]);
		letterCombinations(digits, dicts, ix + 1, ind, res); // here use "ix + 1" because of this problem
		ind.pop_back(); // BACKTRACKING
	}
}

vector<string> letterCombinations(string digits) {
	vector<string> res;
	vector<string> dicts{ "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
	letterCombinations(digits, dicts, 0, "", res);
	return res;
}

/* 78. Subsets */
/* Given a set of distinct integers, nums, return all possible subsets (the power set).
* Note: The solution set must not contain duplicate subsets. */
void subsets(vector<int>& nums, int ix, vector<int>& visited, vector<int>& ind, vector<vector<int>>& res) {
	res.push_back(ind);
	for (int i = ix; i < nums.size(); ++i) {
		if (!visited[i]) {
			visited[i] = 1;
			ind.push_back(nums[i]);
			subsets(nums, i + 1, visited, ind, res);
			ind.pop_back();
			visited[i] = 0;
		}
	}
}
vector<vector<int>> subsets(vector<int>& nums) {
	vector<vector<int>> res;
	int n = nums.size();
	vector<int> visited(n, 0);
	vector<int> ind;
	subsets(nums, 0, visited, ind, res);
	return res;
}

/* 90. Subsets II */
/* Given a collection of integers that might contain duplicates, nums, 
* return all possible subsets (the power set).
* Note: The solution set must not contain duplicate subsets. */
void subsetsWithDup(vector<int>& nums, int ix, vector<int>& visited, vector<int>& ind, vector<vector<int>>& res) {
	res.push_back(ind);
	for (int i = ix; i < nums.size(); ++i) {
		if (!visited[i]) {
			if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]) continue;
			visited[i] = 1;
			ind.push_back(nums[i]);
			subsetsWithDup(nums, i + 1, visited, ind, res);
			ind.pop_back();
			visited[i] = 0;
		}
	}
}

vector<vector<int>> subsetsWithDup(vector<int>& nums) {
	vector<vector<int>> res;
	int n = nums.size();
	sort(nums.begin(), nums.end());
	vector<int> visited(n, 0);
	vector<int> ind;
	subsetsWithDup(nums, 0, visited, ind, res);
	return res;
}



/* ======================== Linked List ========================== */
/* 141. Linked List Cycle */
bool hasCycle(ListNode *head) {
	if (!head) return false;
	ListNode* slow = head, *fast = head;
	while (fast->next && fast->next->next) {
		slow = slow->next;
		fast = fast->next->next;
		if (slow == fast) return true;
	}
	return false;
}

/* 142. Linked List Cycle II */
ListNode *detectCycle(ListNode *head) {
	if (!head) return NULL;
	ListNode* slow = head, *fast = head;
	while (fast->next && fast->next->next) {
		slow = slow->next;
		fast = fast->next->next;
		if (slow == fast) break; // IMPORTANT!
	}
	// boundary condition
	if (!fast->next || !fast->next->next) return NULL;
	fast = head;

	while (slow != fast) {
		slow = slow->next;
		fast = fast->next;
	}
	return slow;
}

/* 206. Reverse Linked List */
/* Example: Input: 1->2->3->4->5->NULL. Output: 5->4->3->2->1->NULL. */
ListNode* reverseList(ListNode* head) {
	if (!head) return NULL;
	ListNode* dummy = new ListNode(-1);
	dummy->next = head;
	ListNode* pre = dummy, *cur = head;
	while (cur && cur->next) {
		ListNode* t = cur->next;
		cur->next = t->next;
		t->next = pre->next;
		pre->next = t;
	}
	return dummy->next;
}

/* 92. Reverse Linked List II */
/* Reverse a linked list from position m to n. Do it in one-pass.
* Note: 1 ≤ m ≤ n ≤ length of list. Input: 1->2->3->4->5->NULL, m = 2, n = 4
* Output: 1->4->3->2->5->NULL */
ListNode* reverseBetween(ListNode* head, int m, int n) {
	if (m < 1 || m >= n || !head) return head;
	ListNode* dummy = new ListNode(-1);
	dummy->next = head;
	ListNode* pre = dummy;

	for (int i = 0; i < m - 1; ++i) pre = pre->next;
	ListNode* cur = pre->next;

	for (int i = 0; i < n - m; ++i) {
		ListNode* t = cur->next;
		cur->next = t->next;
		t->next = pre->next;
		pre->next = t;
	}
	return dummy->next;
}

// 2. Add two numbers
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
	ListNode* res = new ListNode(-1), *pre = res;
	int carry = 0;

	while (l1 || l2) {
		int a = l1 ? l1->val : 0;
		int b = l2 ? l2->val : 0;
		int sum = a + b + carry;

		ListNode* newnode = new ListNode(sum % 10);
		carry = sum / 10;

		pre->next = newnode;
		pre = newnode;
		if (l1) l1 = l1->next;  // IMPORTANT: Not forget "if" condition
		if (l2) l2 = l2->next;
	}
	if (carry) pre->next = new ListNode(carry);
	return res->next;
}

// 445. Add Two Numbers II
ListNode* addTwoNumbers2(ListNode* l1, ListNode* l2) {
	stack<int> st1, st2;
	while (l1) {
		st1.push(l1->val);
		l1 = l1->next;
	}

	while (l2) {
		st2.push(l2->val);
		l2 = l2->next;
	}

	ListNode* res = new ListNode(-1);
	int carry = 0;
	while (!st1.empty() || !st2.empty()) {
		int sum = 0;
		sum += carry;
		if (!st1.empty()) {
			sum += st1.top(); st1.pop();
		}
		if (!st2.empty()) {
			sum += st2.top(); st2.pop();
		}

		res->val = sum % 10;
		ListNode* newnode = new ListNode(sum / 10);
		carry = sum / 10;

		newnode->next = res;
		res = newnode;
	}
	if (carry) res->val = carry;
	return carry ? res : res->next;
}

/* 114. Flatten Binary Tree to Linked List */
/* Given a binary tree, flatten it to a linked list in-place. */
void flattenBTtoLL(TreeNode* root) {
	if (!root) return;
	TreeNode* cur = root;
	if (cur->left) flattenBTtoLL(cur->left);
	if (cur->right) flattenBTtoLL(cur->right);
	TreeNode* t = cur->right;
	cur->right = cur->left;
	cur->left = NULL;
	while (cur->right) cur = cur->right;
	cur->right = t;
	t = NULL;
}

/* 430. Flatten a Multilevel Doubly Linked List */
ListNodeMultiLevel* flattenMultiLL(ListNodeMultiLevel* head) {
	if (!head) return NULL;
	ListNodeMultiLevel* cur = head;
	while (cur) {
		if (cur->child) {
			cur->child = flattenMultiLL(cur->child);
			ListNodeMultiLevel* t = cur->next;
			cur->next = cur->child;
			cur->child->prev = cur; // IMPORTANT
			cur->child = NULL;

			while (cur->next) cur = cur->next;
			cur->next = t;
			if (t) t->prev = cur; // IMPORTANT
			t = NULL;
		}
		cur = cur->next;
	}
	return head;
}

/* 21. Merge Two Sorted Lists */
/* Merge two sorted linked lists and return it as a new list.
* The new list should be made by splicing together the nodes of the first two lists.
* Input: 1->2->4, 1->3->4. Output: 1->1->2->3->4->4.  */
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	if (!l1 || !l2) return l1 ? l1 : l2;
	if (l1->val < l2->val) {
		l1->next = mergeTwoLists(l1->next, l2);
		return l1;
	}
	else {
		l2->next = mergeTwoLists(l1, l2->next);
		return l2;
	}
}

/* 23. Merge k Sorted Lists */
/* Input: [ 1->4->5, 1->3->4, 2->6 ]. Output: 1->1->2->3->4->4->5->6  */
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	if (!l1 || !l2) return l1 ? l1 : l2;
	if (l1->val < l2->val) {
		l1->next = mergeTwoLists(l1->next, l2);
		return l1;
	}
	else {
		l2->next = mergeTwoLists(l1, l2->next);
		return l2;
	}
}

ListNode* mergeKLists(vector<ListNode*>& lists) {
	if (lists.empty()) return NULL;
	int n = lists.size();
	while (n > 1) {
		int k = (n + 1) / 2;
		for (int i = 0; i < n / 2; ++i) {
			lists[i] = mergeTwoLists(lists[i], lists[i + k]);
		}
		n = k;
	}
	return lists[0];
}

/* Remove Linked List Elements */
/* Remove all elements from a linked list of integers that have value val.
* Input:  1->2->6->3->4->5->6, val = 6. Output: 1->2->3->4->5 */
ListNode* removeElements(ListNode* head, int val) {
	if (!head) return NULL;
	ListNode* dummy = new ListNode(-1), *pre = dummy;
	dummy->next = head;

	while (pre->next) {
		ListNode* cur = pre->next;
		if (cur->val == val) {
			pre->next = cur->next;
		}
		else {
			pre = pre->next;
		}
	}
	return dummy->next;
}

/* Remove Duplicates from Sorted List II */
/* Given a sorted linked list, delete all nodes that have duplicate numbers,
* leaving only distinct numbers from the original list. Input: 1->2->3->3->4->4->5. Output: 1->2->5 */
ListNode* deleteDuplicates(ListNode* head) {
	if (!head) return NULL;
	ListNode* dummy = new ListNode(-1), *pre = dummy;
	dummy->next = head;

	while (pre->next) {
		ListNode* cur = pre->next;
		while (cur->next && cur->val == cur->next->val) {
			cur = cur->next;
		}
		if (pre->next != cur) pre->next = cur->next;
		else pre = pre->next;
	}
	return dummy->next;
}

/* 426. Convert Binary Search Tree to Sorted Doubly Linked List */
/* Convert a BST to a sorted circular doubly-linked list in-place.
* Think of the left and right pointers as synonymous to the
* previous and next pointers in a doubly-linked list. */
CircularNode* treeToDoublyList(CircularNode* root) {
	if (!root) return NULL;
	CircularNode* head = NULL, *pre = NULL;
	treeToDoublyList(root, head, pre);
	pre->right = head;
	head->left = pre;
	return head;
}

void treeToDoublyList(CircularNode* node, CircularNode*& head, CircularNode*& pre) {
	if (!node) return;
	treeToDoublyList(node->left, head, pre);
	if (!head) {
		head = node;
		pre = node;
	}
	else {
		pre->right = node;
		node->left = pre;
		pre = node;
	}
	treeToDoublyList(node->right, head, pre);
}

/* 138. Copy List with Random Pointer
* A linked list is given such that each node contains an additional
* random pointer which could point to any node in the list or null.
* Return a deep copy of the list.*/
RandomNode* copyRandomList(RandomNode* head) {
	if (!head) return NULL;
	unordered_map<RandomNode*, RandomNode*> m;
	RandomNode* res = new RandomNode();
	res->val = head->val;
	m[head] = res;
	RandomNode* cur = head->next, *node = res;

	while (cur) {
		RandomNode* t = new RandomNode();
		t->val = cur->val;
		node->next = t;
		m[cur] = t;

		cur = cur->next;
		node = node->next;
	}

	cur = head, node = res;

	while (cur) {
		node->random = m[cur->random];
		cur = cur->next;
		node = node->next;
	}
	return res;
}

/* 147. Insertion Sort List */
ListNode* insertionSortList(ListNode* head) {
	ListNode* dummy = new ListNode(-1);
	dummy->next = head;
	ListNode* pre = dummy, *cur = head;

	while (cur) {
		if (cur->next && cur->val > cur->next->val) {
			while (pre->next && pre->next->val < cur->next->val) {
				pre = pre->next;
			}
			ListNode* t = pre->next;
			pre->next = cur->next;
			cur->next = cur->next->next;
			pre->next->next = t;
			pre = dummy; // reset pre to dummy node
		}
		else {
			cur = cur->next;
		}
	}
	return dummy->next;
}

/* 160. Intersection of Two Linked Lists */
int getLen(ListNode* head) {
	int res = 0;
	while (head) {
		++res;
		head = head->next;
	}
	return res;
}

/* 328. Odd Even Linked List -- PRACTICE
* Input: 1->2->3->4->5->NULL
* Output: 1->3->5->2->4->NULL
* Input: 2->1->3->5->6->4->7->NULL
* Output: 2->3->6->7->1->5->4->NULL
* You should try to do it in place.
* The program should run in O(1) space complexity
* and O(nodes) time complexity.*/
ListNode* oddEvenList(ListNode* head) {
	if (!head) return NULL;
	ListNode* pre = head, *cur = head->next;
	while (cur && cur->next) {
		ListNode* t = pre->next;
		pre->next = cur->next;
		cur->next = cur->next->next;
		pre->next->next = t;
		pre = pre->next;
		cur = cur->next;
	}
	return head;
}

/* 234. Palindrome Linked List
* Input: 1->2->2->1. Output: true. Input: 1->2->3->2->1. Output: true */
bool isPalindrome(ListNode* head) {
	if (!head) return true;
	stack<int> st;
	st.push(head->val);
	ListNode* slow = head, *fast = head;
	while (fast->next && fast->next->next) {
		slow = slow->next;
		fast = fast->next->next;
		st.push(slow->val);
	}

	if (!fast->next) st.pop();

	fast = slow->next;
	// IMPORTANT: DO not forgot about the "!st.empty()"
	while (!st.empty() && slow != fast) {
		auto t = st.top();
		if (t != fast->val) return false;
		fast = fast->next;
		st.pop();
	}
	return true;
}

/* 19. Remove Nth Node From End of List */
/* Given a linked list, remove the n-th node from the end of list and return its head.
* Given: 1->2->3->4->5, and n = 2. => 1->2->3->5. Could you do this in one pass? */
ListNode* removeNthFromEnd(ListNode* head, int n) {
	ListNode* p1 = head, *p2 = head;
	for (int i = 0; i < n; ++i) {
		p1 = p1->next;
	}
	if (!p1->next) return head->next;
	while (p1->next) {
		p1 = p1->next;
		p2 = p2->next;
	}
	p2->next = p2->next->next;
	return head;
}

/* 24. Swap Nodes in Pairs */
ListNode* swapPairs(ListNode* head) {
	if (!head) return NULL;
	ListNode* dummy = new ListNode(-1);
	dummy->next = head;
	ListNode* pre = dummy, *cur = head;
	while (cur && cur->next) {
		ListNode* t = cur->next;
		cur->next = t->next;
		t->next = pre->next;
		pre->next = t;

		pre = cur;
		cur = cur->next;
	}
	return dummy->next;
}




// ======================== Backtracking ========================== */
/* 22. Generate Parentheses */
/* Given n pairs of parentheses, write a function to generate all combinations of
* well-formed parentheses. */
void generateParenthesis(int left, int right, string ind, vector<string>& res) {
	if (left > right) return;
	if (left == 0 && right == 0) {
		res.push_back(ind);
	}
	if (left) generateParenthesis(left - 1, right, ind + "(", res);
	if (right) generateParenthesis(left, right - 1, ind + ")", res);
}

vector<string> generateParenthesis(int n) {
	vector<string> res;
	generateParenthesis(n, n, "", res);
	return res;
}

/* 46. Permutations. Given a collection of distinct integers, return all possible permutations. */
void permute(vector<int>& nums, vector<int>& visited, int ix, vector<int>& ind, vector<vector<int>>& res) {
	if (ix == nums.size()) {
		res.push_back(ind);
		return;
	}
	for (int i = 0; i < nums.size(); ++i) {
		if (!visited[i]) {
			visited[i] = 1;
			ind.push_back(nums[i]);
			permute(nums, visited, ix + 1, ind, res);
			ind.pop_back();
			visited[i] = 0;
		}
	}
}

vector<vector<int>> permute(vector<int>& nums) {
	vector<vector<int>> res;
	if (nums.empty()) return res;
	vector<int> ind;
	sort(nums.begin(), nums.end());
	vector<int> visited(nums.size(), 0);
	permute(nums, visited, 0, ind, res);
	return res;
}


/* ======================== Hash map ============================= */
/* 739. Daily Temperatures */
/* Given a list of daily temperatures T, return a list such that, for each day in the input,
* tells you how many days you would have to wait until a warmer temperature. If there is no
* future day for which this is possible, put 0 instead.
* For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73],
* your output should be [1, 1, 4, 2, 1, 1, 0, 0]. */
vector<int> dailyTemperatures(vector<int>& T) {
	int n = T.size();
	vector<int> res(n, 0);
	stack<int> st;

	for (int i = 0; i < n; ++i) {
		while (!st.empty() && T[i] > T[st.top()]) {
			res[st.top()] = i - st.top();
			st.pop();
		}
		st.push(i);
	}
	return res;
}

/* 387. First Unique Character in a String */
int firstUniqChar(string s) {
	unordered_map<char, int> m;
	for (auto c : s) ++m[c];

	for (int i = 0; i < s.size(); ++i) {
		if (m[s[i]] == 1) return i;
	}
	return -1;
}

/* 49. Group Anagrams */
/* Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output: [["ate","eat","tea"], ["nat","tan"], ["bat"]] */
vector<vector<string>> groupAnagrams(vector<string>& strs) {
	vector<vector<string>> res;
	unordered_map<string, vector<string>> m;
	for (auto s : strs) {
		string t(s);
		sort(t.begin(), t.end());
		m[t].push_back(s);
	}
	for (auto it : m) {
		res.push_back(it.second);
	}
	return res;
}

/* 846. Hand of Straights */
/* Alice has a hand of cards, given as an array of integers. Now she wants to rearrange
* the cards into groups so that each group is size W, and consists of W consecutive cards.
* Return true if and only if she can. */
bool isNStraightHand(vector<int>& hand, int W) {
	map<int, int> m;
	if (hand.size() % W != 0) return false;
	for (auto a : hand) ++m[a];
	for (auto it : m) {
		if (it.second > 0) {
			for (int i = W - 1; i >= 0; --i) {
				if (m[it.first + i] -= m[it.first] < 0) return false;
			}
		}
	}
	return true;
}

/* 202. Happy Number */
/* Write an algorithm to determine if a number is "happy".
* A happy number is a number defined by the following process: Starting with any positive integer,
* replace the number by the sum of the squares of its digits, and repeat the process until the
* number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
* Those numbers for which this process ends in 1 are happy numbers. Eg: 19 -> true. */
int isHappyCal(int n) {
	int res = 0;
	while (n > 0) {
		res += (n % 10) * (n % 10);
		n /= 10;
	}
	return res;
}

bool isHappy(int n) {
	if (n < 1) return false;
	if (n == 1) return true;
	set<int> st;
	while (n != 1) {
		int t = isHappyCal(n);
		if (t == 1) return true;
		if (st.count(t)) return false;
		st.insert(t);
		n = t;
	}
	return true;
}

/* 3. Longest Substring Without Repeating Characters */
/* Given a string, find the length of the longest substring without repeating characters.
* Input: "abcabcbb". Output: 3. Explanation: The answer is "abc", with the length of 3.
* Input: "pwwkew". Output: 3. Explanation: The answer is "wke", with the length of 3. */
int lengthOfLongestSubstring(string s) {
	int res = 0, left = 0, n = s.size();
	unordered_map<char, int> m;

	for (int i = 0; i < n; ++i) {
		// update res if: (1) a new char, (2) 
		if (m[s[i]] == 0 || left > m[s[i]]) {
			res = max(res, i - left + 1);
		}
		else {
			left = m[s[i]];
		}
		m[s[i]] = i + 1;
	}
	return res;
}

/* Subarray Sum Equals K */
/* Given an array of integers and an integer k,
* you need to find the total number of continuous subarrays whose sum equals to k.
* Input:nums = [1,1,1], k = 2. Output: 2. */
int subarraySum(vector<int>& nums, int k) {
	int res = 0, sum = 0;
	unordered_map<int, int> m;
	for (int i = 0; i < nums.size(); ++i) {
		sum += nums[i];
		if (sum == k) ++res;
		res += m[sum - k];
		++m[sum];
	}
	return res;
}

/* Two sum */
vector<int> twoSum(vector<int>& nums, int target) {
	unordered_map<int, int> m;
	for (int i = 0; i < nums.size(); ++i) {
		if (m.find(target - nums[i]) != m.end()) return { i, m[target - nums[i]] };
		m[nums[i]] = i;
	}
	return { -1, -1 };
}

/* Valid Anagram */
bool isAnagram(string s, string p) {
	if (s.size() != p.size()) return false; // IMPORTANT!
	unordered_map<char, int> m;
	for (auto c : s) ++m[c];
	for (auto c : p) {
		if (--m[c] < 0 || !m.count(c)) return false;
	}
	return true;
}

/* Valid Parentheses */
bool isValidParentheses(string s) {
	stack<char> st;
	for (auto c : s) {
		if (c == '(' || c == '[' || c == '{') st.push(c);
		else {
			if (st.empty()) return false;
			else if (c == ')' && st.top() != '(') return false;
			else if (c == ']' && st.top() != '[') return false;
			else if (c == '}' && st.top() != '{') return false;
			st.pop();
		}
	}
	return st.empty();
}

// Integer to romans
string intToRoman(int num) {
	vector<string> strs{ "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
	vector<int> nums{ 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
	string res("");

	for (int i = 0; i < strs.size(); ++i) {
		// while loop is used at this level. Think about it. 
		while (num >= nums[i]) {
			res += strs[i];
			num -= nums[i];
		}
	}
	return res;
}

/* Longest Consecutive Sequence */
/* Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
* Your algorithm should run in O(n) complexity. [100, 4, 200, 1, 3, 2] */
int longestConsecutive(vector<int>& nums) {
	if (nums.empty()) return 0;
	unordered_set<int> st(nums.begin(), nums.end());
	int res = 0;
	for (int i = 0; i < nums.size(); ++i) {
		int left = nums[i] - 1, right = nums[i] + 1;
		while (st.count(left)) st.erase(left--);
		while (st.count(right)) st.erase(right++);

		res = max(res, right - left - 1);
	}
	return res;
}

/* 554. Brick Wall */
/* There is a brick wall in front of you. The wall is rectangular and has several rows of bricks. 
* The bricks have the same height but different width. You want to draw a vertical line from
* the top to the bottom and cross the least bricks. If your line go through the edge of a brick, 
* then the brick is not considered as crossed. You need to find out how to draw the line to cross 
* the least bricks and return the number of crossed bricks. 
* Input: [[1,2,2,1],
        [3,1,2],
        [1,3,2],
        [2,4],
        [3,1,2],
        [1,3,1,1]]. Output: 2 */
int leastBricks(vector<vector<int>>& wall) {
	int mx = 0;
	unordered_map<int, int> m;
	for (auto a : wall) {
		int sum = 0;
		for (int i = 0; i < a.size() - 1; ++i) {
			sum += a[i];
			++m[sum];
			mx = max(mx, m[sum]);
		}
	}
	return wall.size() - mx;
}

/* 438. Find All Anagrams in a String */
/* Given a string s and a non-empty string p, find all the start indices of p's
* anagrams in s. Strings consists of lowercase English letters only and the
* length of both strings s and p will not be larger than 20,100.
* Input: s: "cbaebabacd" p: "abc". Output: [0, 6]. */
vector<int> findAnagrams(string s, string p) {
	vector<int> res, v(128, 0);
	for (auto c : p) ++v[c];
	int i = 0, m = s.size(), n = p.size();
	if (m < n) return res;

	while (i < m) {
		vector<int> t = v;
		bool find = true;
		for (int j = i; j < i + n; ++j) {
			if (--t[s[j]] < 0) {
				find = false;
				break;
			}
		}
		if (find) {
			res.push_back(i);
		}
		++i;
	}
	return res;
}

/* 652. Find Duplicate Subtrees */
/* Given a binary tree, return all duplicate subtrees. For each kind of duplicate
* subtrees, you only need to return the root node of any one of them.
* Logic: postorder traversal and hash map. */
string findDuplicateSubtrees(TreeNode* node, unordered_map<string, int>& m, vector<TreeNode* >& res) {
	if (!node) return "";
	string s = findDuplicateSubtrees(node->left, m, res) + "_" +
		findDuplicateSubtrees(node->right, m, res) + "_" + to_string(node->val);
	if (m[s] == 1) res.push_back(node);
	++m[s];
	return s;
}

vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
	unordered_map<string, int> m;
	vector<TreeNode* > res;
	findDuplicateSubtrees(root, m, res);
	return res;
}

/* 41. First Missing Positive */
/* Given an unsorted integer array, find the smallest missing positive integer.
* Input: [1,2,0]. Output: 3. Input: [7,8,9,11,12]. Output: 1.
* Note: Your algorithm should run in O(n) time and uses constant extra space.*/
int firstMissingPositive(vector<int>& nums) {
	set<int> st(nums.begin(), nums.end());
	int mx = INT_MIN;
	for (auto a : st) {
		mx = max(mx, a);
	}
	for (int i = 1; i <= mx; ++i) {
		if (!st.count(i)) return i;
	}
	return mx > 0 ? mx + 1 : 1;
}

/* 350. Intersection of Two Arrays II */
/* Given two arrays, write a function to compute their intersection.
* Example 1: Input: nums1 = [1,2,2,1], nums2 = [2,2]. Output: [2,2]. */
vector<int> intersect2(vector<int>& nums1, vector<int>& nums2) {
	vector<int>res;
	unordered_map<int, int> m;
	for (auto a : nums1) ++m[a];
	for (auto a : nums2) {
		if (m[a]-- > 0) res.push_back(a);
	}
	return res;
}

/* 205. Isomorphic Strings */
/* Given two strings s and t, determine if they are isomorphic.
* Two strings are isomorphic if the characters in s can be replaced to get t.
* Input: s = "egg", t = "add". Output: true. */
bool isIsomorphic(string s, string t) {
	if (s.size() != t.size()) return false;
	int n = s.size();
	vector<int> v1, v2;
	for (int i = 0; i < n; ++i) {
		if (v1[s[i]] != v2[t[i]]) return false;
		v1[s[i]] = i + 1;
		v2[t[i]] = i + 1;
	}
	return true;
}

/* 582. Kill Process */
vector<int> killProcess(vector<int>& pid, vector<int>& ppid, int kill) {
	vector<int> res;
	unordered_map<int, vector<int>> m;
	int n = pid.size();
	for (int i = 0; i < n; ++i) {
		m[ppid[i]].push_back(pid[i]);
	}
	queue<int> q{ { kill } };
	while (!q.empty()) {
		auto t = q.front(); q.pop();
		res.push_back(t);
		for (auto it : m[t]) {
			q.push(it);
		}
	}
	return res;
}

/* 84. Largest Rectangle in Histogram */
/* Given n non-negative integers representing the histogram's bar height
* where the width of each bar is 1, find the area of largest rectangle in the histogram.
* Input: [2,1,5,6,2,3]. Output: 10. */
int largestRectangleArea(vector<int>& heights) {
	heights.push_back(0);
	int n = heights.size(), res = 0;
	stack<int> st;
	for (int i = 0; i < n; ++i) {
		while (!st.empty() && heights[i] <= heights[st.top()]) {
			auto t = st.top(); st.pop();
			res = max(res, heights[t] * (st.empty() ? i : (i - st.top() - 1)));
		}
		st.push(i);
	}
	return res;
}

/* 340. Longest Substring with At Most K Distinct Characters */
/* Given a string, find the length of the longest substring T that contains at most k distinct characters.
*  Input: s = "eceba", k = 2. Output: 3. Explanation: T is "ece" which its length is 3.*/
int lengthOfLongestSubstringKDistinct(string s, int k) {
	int n = s.size(), res = 0, left = 0;
	unordered_map<char, int> m;
	for (int i = 0; i < n; ++i) {
		++m[s[i]];
		while (m.size() > k) {
			if (--m[s[i]] == 0) m.erase(s[i]);
			++left;
		}
		res = max(res, i - left + 1);
	}
	return res;
}

/* 524. Longest Word in Dictionary through Deleting */
/* Input: s = "abpcplea", d = ["ale","apple","monkey","plea"]. Output: "apple" */
string findLongestWord(string s, vector<string>& d) {
	sort(d.begin(), d.end(), [](string s, string p) {
		return s.size() > p.size() || (s.size() == p.size() && s < p);
	});
	int n = s.size();
	for (auto str : d) {
		int i = 0, j = 0;
		while (i < str.size() && j < n) {
			if (str[i] == s[j]) ++i;
			++j;
		}
		if (i == str.size()) return str;
	}
	return "";
}

/* 149. Max Points on a Line */
/* Given n points on a 2D plane, find the maximum number of points that lie on the same straight line. */
int gcd(int a, int b) {
	return b == 0 ? a : gcd(b, a % b);
}

int maxPoints(vector<vector<int>>& points) {
	int res = 0, n = points.size();

	for (int i = 0; i < n; ++i) {
		int dup = 1;
		map<pair<int, int>, int> m;

		for (int j = i + 1; j < n; ++j) {
			if (points[i][0] == points[j][0] && points[i][1] == points[j][1]) {
				++dup;
				continue;
			}
			int dx = points[j][0] - points[i][0];
			int dy = points[j][1] - points[i][1];
			int d = gcd(dx, dy);
			++m[{dx / d, dy / d}];
		}
		res = max(res, dup);
		for (auto it : m) {
			res = max(res, it.second + dup);
		}
	}
	return res;
}

/* 266. Palindrome Permutation */
/* Given a string, determine if a permutation of the string could form a palindrome.
* Input: "aab". Output: true. */
bool canPermutePalindrome(string s) {
	unordered_map<char, int> m;
	int odd = 0, n = s.size();
	for (auto c : s) ++m[c];
	for (auto it : m) {
		if (it.second % 2 == 1)  ++odd;
	}
	return (odd % 2 == 0 && n % 2 == 0) || (odd % 2 == 1 && n % 2 == 1);
}

/* 567. Permutation in String */
/* Given two strings s1 and s2, write a function to return true if s2 contains the permutation of s1.
* In other words, one of the first string's permutations is the substring of the second string.
* Input: s1 = "ab" s2 = "eidbaooo". Output: True. */
// Solution 1: Brute forces, OTL.
bool isPermutation(string s, string p) {
	if (s.size() != p.size()) return false;
	unordered_map<char, int> m;
	for (auto c : s) ++m[c];
	for (auto c : p) {
		if (--m[c] < 0) return false;
	}
	return true;
}

bool checkInclusion(string s1, string s2) {
	int m = s1.size(), n = s2.size();
	if (m > n) return false;
	for (int i = 0; i <= (n - m); ++i) {

		if (isPermutation(s1, s2.substr(i, m))) {
			//cout << s2.substr(i, m) << endl; 
			return true;
		}
	}
	return false;
}




// =============================== Queue ========================== */
/* 239. Sliding Window Maximum */
/* Given an array nums, there is a sliding window of size k which is moving from the very left of the array
* to the very right. You can only see the k numbers in the window. Each time the sliding window moves
* right by one position. Return the max sliding window. */
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
	deque<int> q;
	vector<int> res;
	int n = nums.size();
	for (int i = 0; i < n; ++i) {
		if (!q.empty() && i - q.front() >= k) q.pop_front();
		while (!q.empty() && nums[i] > nums[q.back()]) q.pop_back();
		q.push_back(i);
		if (i >= k - 1) res.push_back(nums[q.front()]);
	}
	return res;
}

/* 451. Sort Characters By Frequency */
/* Given a string, sort it in decreasing order based on the frequency of characters.
* Input: "tree". Output:  "eert". */
string frequencySort(string s) {
	string res("");
	unordered_map<char, int> m;
	priority_queue<pair<int, char>> q;
	for (auto c : s) ++m[c];
	for (auto it : m) {
		q.push({ it.second, it.first });
	}
	while (!q.empty()) {
		auto t = q.top(); q.pop();
		res += string(t.first, t.second);
	}
	return res;
}

/* 347. Top K Frequent Elements */
/* Given a non-empty array of integers, return the k most frequent elements.
* Example 1: Input: nums = [1,1,1,2,2,3], k = 2. Output: [1,2]. */
vector<int> topKFrequent(vector<int>& nums, int k) {
	vector<int> res;
	unordered_map<int, int> m;
	priority_queue<pair<int, int>> q;
	for (auto a : nums) ++m[a];
	for (auto it : m) q.push({ it.second, it.first });

	for (int i = 0; i < k; ++i) {
		auto t = q.top(); q.pop();
		res.push_back(t.second);
	}
	return res;
}

/* 692. Top K Frequent Words */
/* Given a non-empty list of words, return the k most frequent elements.
* Your answer should be sorted by frequency from highest to lowest.
* If two words have the same frequency, then the word with the lower
* alphabetical order comes first. */
vector<string> topKFrequent(vector<string>& words, int k) {
	vector<string> res;
	unordered_map<string, int> m;
	for (auto s : words) ++m[s];

	auto comp = [](pair<int, string>& a, pair<int, string>& b) {
		return a.first < b.first || (a.first == b.first && a.second > b.second);
	};
	priority_queue<pair<int, string>, vector<pair<int, string>>, decltype(comp)> q(comp);
	for (auto it : m) q.push({ it.second, it.first });

	for (int i = 0; i < k; ++i) {
		auto t = q.top(); q.pop();
		res.push_back(t.second);
	}
	return res;
}


/* ========================== Array Problems ============================ */
/* 169. Majority Element */
/* Given an array of size n, find the majority element. The majority element is
* the element that appears more than ⌊ n/2 ⌋ times. Input: [2,2,1,1,1,2,2]. Output: 2. */
int majorityElement(vector<int>& nums) {
	int cnt = 0, res = nums[0];
	for (auto a : nums) {
		if (a == res) {
			++cnt;
		}
		else {
			if (cnt) --cnt;
			else res = a;
		}
	}
	return res;
}

/* 229. Majority Element II */
/* Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.
* Note: The algorithm should run in linear time and in O(1) space. Input: [1,1,1,3,3,2,2,2]. Output: [1,2]. */
vector<int> majorityElement2(vector<int>& nums) {
	vector<int> res;
	int m = 0, n = 0, cm = 0, cn = 0, len = nums.size();
	for (auto a : nums) {
		if (a == m) ++cm;
		else if (a == n) ++cn;
		else if (cm == 0) { m = a; ++cm; }
		else if (cn == 0) { n = a; ++cn; }
		else { --cm; --cn; }
	}
	cm = 0, cn = 0;
	for (auto a : nums) {
		if (a == m) ++cm;
		else if (a == n) ++cn;
	}
	if (len < cm * 3) res.push_back(m);
	if (len < cn * 3) res.push_back(n);
	return res;
}

/* 283. Move Zeroes */
/* Input: [0,1,0,3,12]. Output: [1,3,12,0,0] */
void moveZeroes(vector<int>& nums) {
	for (int i = 0, j = 0; i < nums.size(); ++i) {
		if (nums[i]) swap(nums[i], nums[j++]);
	}
}

/* 53. Maximum Subarray */
/* Given an integer array nums, find the contiguous subarray (containing at least one number)
* which has the largest sum and return its sum. Input: [-2,1,-3,4,-1,2,1,-5,4],
* Output: 6. Explanation: [4,-1,2,1] has the largest sum = 6. */
int maxSubArray(vector<int>& nums) {
	int res = INT_MIN, sum = 0;
	for (auto a : nums) {
		sum = max(sum + a, a);
		res = max(res, sum);
	}
	return res;
}

/* 4. Median of Two Sorted Arrays */
/* There are two sorted arrays nums1 and nums2 of size m and n respectively.
* Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
* You may assume nums1 and nums2 cannot be both empty. */
double findMedianSortedArrays(vector<int> nums1, vector<int> nums2, int k) {
	int m = nums1.size(), n = nums2.size();
	if (m > n) return findMedianSortedArrays(nums2, nums1, k);
	if (m == 0) return nums2[k - 1];
	if (k == 1) return min(nums1[0], nums2[0]);
	int i = min(m, k / 2), j = min(n, k / 2);
	if (nums1[i - 1] < nums2[j - 1]) {
		return findMedianSortedArrays(vector<int>(nums1.begin() + i, nums1.end()), nums2, k - i);
	}
	else {
		return findMedianSortedArrays(nums1, vector<int>(nums2.begin() + j, nums2.end()), k - j);
	}
}

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
	int m = nums1.size(), n = nums2.size();
	return (findMedianSortedArrays(nums1, nums2, (m + n + 1) / 2) +
		findMedianSortedArrays(nums1, nums2, (m + n + 2) / 2)) / 2;
}

/* 252. Meeting Rooms */
/* Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei),
* determine if a person could attend all meetings. Input: [[0,30],[5,10],[15,20]]. Output: false. */
bool canAttendMeetings(vector<vector<int>>& intervals) {
	if (intervals.empty()) return true;
	sort(intervals.begin(), intervals.end(), [](vector<int>& a, vector<int>& b) {
		return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
	});
	int pre = intervals[0][1];
	for (int i = 1; i < intervals.size(); ++i) {
		if (intervals[i][0] < pre) return false;
		pre = intervals[i][1];
	}
	return true;
}

/* 253. Meeting Rooms II */
/* Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei),
* find the minimum number of conference rooms required. Input: [[0, 30],[5, 10],[15, 20]]. Output: 2. */
int minMeetingRooms(vector<vector<int>>& intervals) {
	int n = intervals.size(), ix = 0, res = 0;
	vector<int> starts(n, 0), ends(n, 0);
	for (int i = 0; i < n; ++i) {
		starts[i] = intervals[i][0];
		ends[i] = intervals[i][1];
	}
	sort(starts.begin(), starts.end());
	sort(ends.begin(), ends.end());

	for (int i = 0; i < n; ++i) {
		if (starts[i] < ends[ix]) ++res;
		else ++ix;
	}
	return res;
}

/* 56. Merge Intervals */
/* Given a collection of intervals, merge all overlapping intervals.
* Input: [[1,3],[2,6],[8,10],[15,18]]. Output: [[1,6],[8,10],[15,18]]
* Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6]. */
vector<vector<int>> merge(vector<vector<int>>& intervals) {
	vector<vector<int>> res;
	if (intervals.empty()) return res;
	sort(intervals.begin(), intervals.end(), [](vector<int>& a, vector<int>& b) {
		return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
	});
	res.push_back(intervals[0]);
	int  n = intervals.size();
	for (int i = 1; i < n; ++i) {
		if (intervals[i][0] <= res.back()[1]) {
			res.back()[1] = max(res.back()[1], intervals[i][1]);
		}
		else {
			res.push_back(intervals[i]);
		}
	}
	return res;
}

/* 1272. Remove Interval */
/* Given a sorted list of disjoint intervals, each interval intervals[i] = [a, b] represents the
* set of real numbers x such that a <= x < b. We remove the intersections between any interval in intervals
* and the interval toBeRemoved. Return a sorted list of intervals after all such removals.
* Input: intervals = [[0,2],[3,4],[5,7]], toBeRemoved = [1,6]. Output: [[0,1],[6,7]].
* Input: intervals = [[0,5]], toBeRemoved = [2,3]. Output: [[0,2],[3,5]]. */
vector<vector<int>> removeInterval(vector<vector<int>>& intervals, vector<int>& toBeRemoved) {
	vector<vector<int>> res;
	int start = toBeRemoved[0], end = toBeRemoved[1];
	for (auto a : intervals) {
		if (a[1] <= start || a[0] >= end) res.push_back(a);
		else {
			if (a[0] < start) res.push_back({ a[0], start });
			if (a[1] > end) res.push_back({ end, a[1] });
		}
	}
	return res;
}

/* 88. Merge Sorted Array */
/* Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
*  The number of elements initialized in nums1 and nums2 are m and n respectively.
* You may assume that nums1 has enough space (size that is greater or equal to m + n) to
* hold additional elements from nums2.
* Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3. Output: [1,2,2,3,5,6]. */
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
	int p1 = m - 1, p2 = n - 1, p3 = m + n - 1;
	while (p1 >= 0 && p2 >= 0) {
		if (nums1[p1] > nums2[p2]) nums1[p3--] = nums1[p1--];
		else nums1[p3--] = nums2[p2--];
	}
	while (p1 >= 0) nums1[p3--] = nums1[p1--];
	while (p2 >= 0) nums1[p3--] = nums2[p2--];
}

/* 836. Rectangle Overlap */
/* A rectangle is represented as a list [x1, y1, x2, y2], where (x1, y1) are the coordinates
* of its bottom-left corner, and (x2, y2) are the coordinates of its top-right corner.
* Two rectangles overlap if the area of their intersection is positive.
* To be clear, two rectangles that only touch at the corner or edges do not overlap. */
bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2) {
	return !(rec1[2] <= rec2[0] || rec1[3] <= rec2[1] ||
		rec2[2] <= rec1[0] || rec2[3] <= rec1[1]);
}

/* Search Insert Position */
/* Given a sorted array and a target value, return the index if the target is found.
* If not, return the index where it would be if it were inserted in order.
* You may assume no duplicates in the array. Input: [1,3,5,6], 5. Output: 2 */
int searchInsert(vector<int>& nums, int target) {
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] >= target) return i;
	}
	return nums.size(); // IMPORTANT.
}

// Reverse Integer
int reverseInteger(int n) {
	long long res = 0;
	while (n != 0) {
		res = res * 10 + n % 10;
		n /= 10;
	}
	return res > INT_MAX || res < INT_MIN ? -1 : res;
}

// Single Number
int singleNumber(vector<int>& nums) {
	int res = 0;
	for (auto a : nums) res ^= a;
	return res;
}

/* 204. Count Primes -- ARRAY & MATH */
/* Count the number of prime numbers less than a non-negative number, n. */
int countPrimes(int n) {
	if (n <= 1) return 0;
	int res = 0;
	vector<int> prime(n - 1, 1);
	prime[0] = 0;

	for (int i = 2; i <= sqrt(n); ++i) {
		if (prime[i - 1]) {
			for (int j = i * i; j < n; j += i) {
				prime[j - 1] = 0;
			}
		}
	}

	for (int i = 0; i < n - 1; ++i) {
		if (prime[i]) ++res;
	}
	return res;
}

/* Triangle */
/* Given a triangle, find the minimum path sum from top to bottom.
* Each step you may move to adjacent numbers on the row below.
*      [2],
*	   [3,4],
*    [6,5,7],
*   [4,1,8,3]  => 11 */
int minimumTotalTriangle(vector<vector<int>>& triangle) {
	vector<int> res = triangle.back();
	int m = triangle.size();
	for (int i = m - 2; i >= 0; --i) {
		for (int j = 0; j <= i; ++j) {
			res[j] = triangle[i][j] + min(res[j], res[j + 1]);
		}
	}
	return res[0];
}

// k-th largest element in an array
int findKthLargest(vector<int>& nums, int k) {
	sort(nums.begin(), nums.end());
	return nums[nums.size() - k];
}

// pascals triangle
vector<vector<int>> generateTriangle(int n) {
	vector<vector<int>> res(n, vector<int>(1));
	for (int i = 0; i < n; ++i) {
		res[i][0] = 1;
		if (i == 0) continue;
		for (int j = 1; j < i; ++j) {
			res[i].push_back(res[i - 1][j - 1] + res[i - 1][j]);
		}
		res[i].push_back(1);
	}

	return res;
}

/* 723. Candy Crush */
/* This question is about implementing a basic elimination algorithm for Candy Crush.
* you need to restore the board to a stable state by crushing candies according to the rule.*/
vector<vector<int>> candyCrush(vector<vector<int>>& board) {
	int m = board.size(), n = board[0].size();
	while (true) {
		vector<pair<int, int>> del;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (board[i][j] == 0) continue;
				int x0 = i, x1 = i, y0 = j, y1 = j;

				while (x0 >= 0 && x0 > i - 3 && board[i][j] == board[x0][j]) --x0;
				while (x1 < m && x1 < i + 3 && board[i][j] == board[x1][j]) ++x1;
				while (y0 >= 0 && y0 > j - 3 && board[i][j] == board[i][y0]) --y0;
				while (y1 < n && y1 < j + 3 && board[i][j] == board[i][y1]) ++y1;

				if (x1 - x0 > 3 || y1 - y0 > 3) del.push_back({ i, j });
			}
		}
		if (del.empty()) break;
		for (auto a : del) board[a.first][a.second] = 0;

		for (int j = 0; j < n; ++j) {
			int t = m - 1;
			for (int i = m - 1; i >= 0; --i) {
				if (board[i][j]) swap(board[i][j], board[t--][j]);
			}
		}
	}
	return board;
}

/* 11. Container With Most Water -- TWO POINTERS */
/* Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai).
* n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0).
* Find two lines, which together with x-axis forms a container, such that the container contains
* the most water. */
int maxArea(vector<int>& height) {
	int res = 0, l = 0, r = height.size() - 1;

	while (l < r) {
		res = max(res, min(height[l], height[r]) * (r - l));
		height[l] < height[r] ? ++l : --r;
	}

	return res;
}

/*172.  Factorial Trailing Zeroes */
/* Given an integer n, return the number of trailing zeroes in n!. */
int trailingZeroes(int n) {
	long long res = 0;
	if (n<0) return 0;
	for (long long i = 5; i <= n; i *= 5) {
		res += (n / i);
	}
	return res;
}

/* 289. Game of Life */
/* Given a board with m by n cells, each cell has an initial state live (1) or dead (0).
* Each cell interacts with its eight neighbors (horizontal, vertical, diagonal)
* using the following four rules (taken from the above Wikipedia article):
* Any live cell with fewer than two live neighbors dies, as if caused by under-population.
* Any live cell with two or three live neighbors lives on to the next generation.
* Any live cell with more than three live neighbors dies, as if by over-population..
* Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
* Write a function to compute the next state (after one update) of the board given its current state.
* The next state is created by applying the above rules simultaneously to every cell in the current state,
* where births and deaths occur simultaneously. */
void gameOfLife(vector<vector<int>>& board) {
	int m = board.size(), n = board[0].size();
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			int cnt = 0;
			for (auto dir : dirs3) {
				int x = i + dir[0], y = j + dir[1];
				if (x >= 0 && x < m && y >= 0 && y < n && (board[x][y] == 1 || board[x][y] == 2)) ++cnt;
			}
			if (board[i][j] && (cnt < 2 || cnt > 3)) board[i][j] = 2;
			else if (!board[i][j] && cnt == 3) board[i][j] = 3;
		}
	}
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			board[i][j] %= 2;
		}
	}
}

/* 274. H-Index */
int hIndex(vector<int>& citations) {
	sort(citations.begin(), citations.end(), greater<int>());
	for (int i = 0; i < citations.size(); ++i) {
		if (i >= citations[i]) return i;
	}
	return citations.size();
}

/* 463. Island Perimeter */
/* You are given a map in form of a two-dimensional integer grid where 1 represents land
* and 0 represents water. The island doesn't have "lakes" (water inside that isn't connected to the water around the island).
* One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100.
* Determine the perimeter of the island. */
int islandPerimeter(vector<vector<int>>& grid) {
	int res = 0, m = grid.size(), n = grid[0].size();
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == 1) {
				if (i == 0 || grid[i - 1][j] == 0) ++res;
				if (i == m - 1 || grid[i + 1][j] == 0) ++res;
				if (j == 0 || grid[i][j - 1] == 0) ++res;
				if (j == n - 1 || grid[i][j + 1] == 0) ++res;
			}
		}
	}
	return res;
}

/* 209. Minimum Size Subarray Sum */
/* Given an array of n positive integers and a positive integer s, find the minimal length of
* a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.
* Example: Input: s = 7, nums = [2,3,1,2,4,3]. Output: 2 */
int minSubArrayLen(int s, vector<int>& nums) {
	int left = 0, n = nums.size(), res = INT_MAX, sum = 0;
	for (int i = 0; i < n; ++i) {
		sum += nums[i];
		while (sum >= s && left <= i) {
			res = min(res, i - left + 1);
			sum -= nums[left++];
		}
	}
	return res == INT_MAX ? 0 : res;
}

/* 496. Next Greater Element I */
/* You are given two arrays (without duplicates) nums1 and nums2 where nums1’s elements
* are subset of nums2. Find all the next greater numbers for nums1's elements in the
* corresponding places of nums2. Input: nums1 = [4,1,2], nums2 = [1,3,4,2].Output: [-1,3,-1]. */
vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
	int n = nums1.size(), m = nums2.size();
	vector<int> res(n, -1);

	for (int i = 0; i < n; ++i) {
		int j = 0;
		for (j = 0; j < m; ++j) {
			if (nums1[i] == nums2[j]) break;
		}
		for (int k = j + 1; k < m; ++k) {
			if (nums2[k] > nums1[i]) {
				res[i] = nums2[k];
				break;
			}
		}
	}
	return res;
}

/* 503. Next Greater Element II */
/* Given a circular array (the next element of the last element is the first element of the array),
* print the Next Greater Number for every element. The Next Greater Number of a number x is the
* first greater number to its traversing-order next in the array, which means you could search
* circularly to find its next greater number. If it doesn't exist, output -1 for this number.
* Input: [1,2,1]. Output: [2,-1,2]. */
vector<int> nextGreaterElements(vector<int>& nums) {
	int n = nums.size();
	vector<int> res(n, -1);
	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < i + n; ++j) {
			if (nums[j % n] > nums[i]) {
				res[i] = nums[j % n];
				break;
			}
		}
	}
	return res;
}

/* 31. Next Permutation */
/* Implement next permutation, which rearranges numbers into the lexicographically next greater
* permutation of numbers. If such arrangement is not possible, it must rearrange it as the
* lowest possible order (ie, sorted in ascending order). The replacement must be in-place and
* use only constant extra memory. Example: [1, 6, 9, 8, 7, 5] -> [1, 7, 5, 6, 8, 9]. */
void nextPermutation(vector<int>& nums) {
	int n = nums.size(), i = n - 1;
	for (; i > 0; --i) {
		if (nums[i] > nums[i - 1]) {
			break;
		}
	}
	if (i == 0) {
		reverse(nums.begin(), nums.end());
		return;
	}
	for (int j = n - 1; j >= i; --j) {
		if (nums[j] > nums[i - 1]) {
			swap(nums[i - 1], nums[j]);
			break;
		}
	}
	reverse(nums.begin() + i, nums.end());
	return;
}

/* 435. Non-overlapping Intervals */
/* Given a collection of intervals, find the minimum number of intervals you need to remove
* to make the rest of the intervals non-overlapping. Input: [[1,2],[2,3],[3,4],[1,3]]. Output: 1. */
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
	sort(intervals.begin(), intervals.end(), [](vector<int>& a, vector<int>& b) {
		return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
	});
	int pre = intervals[0][1], res = 0;
	for (int i = 1; i < intervals.size(); ++i) {
		if (intervals[i][0] < pre) {
			++res;
			pre = min(pre, intervals[i][1]); // IMPORTANT. "min" 
		}
		else {
			pre = intervals[i][1];
		}
	}
	return res;
}

/* 118. Pascal's Triangle */
/* Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.*/
vector<vector<int>> generate(int numRows) {
	vector<vector<int>> res(numRows, vector<int>());
	for (int i = 0; i < numRows; ++i) {
		res[i][0] = 1;
		if (i == 0) continue;
		for (int j = 1; j < i; ++j) {
			res[i][j] = res[i - 1][j - 1] + res[i - 1][j];
		}
		res[i].push_back(1);
	}
	return res;
}

/* 119. Pascal's Triangle II */
/* Given a non-negative index k where k ≤ 33, return the kth index row of the Pascal's triangle.
* Note that the row index starts from 0. Input: 3. Output: [1,3,3,1]. */
vector<int> getRow(int rowIndex) {
	vector<int> res(rowIndex + 1, 1);
	for (int i = rowIndex; i >= 1; --i) {
		for (int j = 1; j < i; ++j) {
			res[j] += res[j - 1];
		}
	}
	return res;
}

/* 238. Product of Array Except Self */
/* Given an array nums of n integers where n > 1,  return an array output such that output[i]
* is equal to the product of all the elements of nums except nums[i].
* Input:  [1,2,3,4]. Output: [24,12,8,6]. */
vector<int> productExceptSelf(vector<int>& nums) {
	int n = nums.size();
	vector<int> res(n, 0), pre(n, 1), post(n, 1);
	for (int i = 1; i < n; ++i) pre[i] = pre[i - 1] * nums[i - 1];
	for (int i = n - 2; i >= 0; --i) post[i] = post[i + 1] * nums[i + 1];
	for (int i = 0; i < n; ++i) res[i] = pre[i] * post[i];
	return res;
}

/* 26. Remove Duplicates from Sorted Array */
/* Given a sorted array nums, remove the duplicates in-place such that each element
* appear only once and return the new length. Do not allocate extra space for another array,
* you must do this by modifying the input array in-place with O(1) extra memory.
* Given nums = [0,0,1,1,1,2,2,3,3,4], Your function should return length = 5. */
int removeDuplicates(vector<int>& nums) {
	if (nums.empty()) return 0;
	int n = nums.size(), j = 0;
	for (int i = 0; i < n; ++i) {
		if (nums[i] != nums[j]) nums[++j] = nums[i];
	}
	return j + 1;
}

/* 80. Remove Duplicates from Sorted Array II */
/* Given a sorted array nums, remove the duplicates in-place such that duplicates
* appeared at most twice and return the new length. Do not allocate extra space for another array,
* you must do this by modifying the input array in-place with O(1) extra memory.
* Given nums = [1,1,1,2,2,3], Your function should return length = 5. */
int removeDuplicates2(vector<int>& nums) {
	int n = nums.size(), i = 0;
	for (auto a : nums) {
		if (i < 2 || a > nums[i - 2]) {
			nums[i++] = a;
		}
	}
	return i;
}

/* 73. Set Matrix Zeroes */
/* Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place. */
void setZeroes(vector<vector<int>>& matrix) {
	int m = matrix.size(), n = matrix[0].size(), row = 0, col = 0;

	for (int i = 0; i < m; ++i) {
		if (matrix[i][0] == 0) row = 1;
	}

	for (int j = 0; j < n; ++j) {
		if (matrix[0][j] == 0) col = 1;
	}

	for (int i = 1; i < m; ++i) {
		for (int j = 1; j < n; ++j) {
			if (matrix[i][j] == 0) {
				matrix[i][0] = 0;
				matrix[0][j] = 0;
			}
		}
	}

	for (int i = 1; i < m; ++i) {
		for (int j = 1; j < n; ++j) {
			if (matrix[i][0] == 0 || matrix[0][j] == 0) {
				matrix[i][j] = 0;
			}
		}
	}

	if (row) {
		for (int i = 0; i < m; ++i) matrix[i][0] = 0;
	}

	if (col) {
		for (int j = 0; j < n; ++j) matrix[0][j] = 0;
	}
}

/* 821. Shortest Distance to a Character */
/* Given a string S and a character C, return an array of integers representing the shortest distance
* from the character C in the string. Input: S = "loveleetcode", C = 'e'.
* Output: [3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0] */
vector<int> shortestToChar(string S, char C) {
	int n = S.size(), pos = -n;
	vector<int> res(n, n);
	for (int i = 0; i < n; ++i) {
		if (S[i] == C) pos = i;
		res[i] = min(res[i], abs(i - pos));
	}
	for (int i = n - 1; i >= 0; --i) {
		if (S[i] == C) pos = i;
		res[i] = min(res[i], abs(pos - i));
	}
	return res;
}

/* 581. Shortest Unsorted Continuous Subarray */
/* Given an integer array, you need to find one continuous subarray that if you only sort this
* subarray in ascending order, then the whole array will be sorted in ascending order, too.
* You need to find the shortest such subarray and output its length. Input: [2, 6, 4, 8, 10, 9, 15]
* Output: 5. */
int findUnsortedSubarray(vector<int>& nums) {
	int n = nums.size(), res = n, start = -1;
	for (int i = 1; i < n; ++i) {
		if (nums[i] < nums[i - 1]) {
			int j = i;
			while (j < n && nums[j] < nums[j - 1]) {
				swap(nums[j], nums[j - 1]);
				--j;
			}
			if (start == -1 || j < start) start = j;
			res = min(res, i - start + 1);
		}
	}
	return res;
}

/* 75. Sort Colors */
/* Given an array with n objects colored red, white or blue, sort them in-place so that
* objects of the same color are adjacent, with the colors in the order red, white and blue.
* Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue 
* respectively. Note: You are not suppose to use the library's sort function for this problem.
* Example: Input: [2,0,2,1,1,0]. Output: [0,0,1,1,2,2]. */
void sortColors(vector<int>& nums) {
	int n = nums.size(), idx = 0;
	vector<int> v(3, 0);
	for (int i = 0; i < n; ++i) {
		++v[nums[i]];
	}
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < v[i]; ++j) {
			nums[idx++] = i;
		}
	}
}

/* 311. Sparse Matrix Multiplication */
vector<vector<int>> multiply(vector<vector<int>>& A, vector<vector<int>>& B) {
	int m = A.size(), k = A[0].size(), n = B[0].size();
	vector<vector<int> > res(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		for (int p = 0; p < k; ++p) {
			if (A[i][p] != 0) {
				for (int j = 0; j < n; ++j) {
					if (B[p][j] != 0) {
						res[i][j] += A[i][p] * B[p][j];
					}
				}
			}
		}
	}
	return res;
}

/* 228. Summary Ranges */
/* Given a sorted integer array without duplicates, return the summary of its ranges.
* Example 1: Input:  [0,1,2,4,5,7]. Output: ["0->2","4->5","7"]
* Explanation: 0,1,2 form a continuous range; 4,5 form a continuous range. */
vector<string> summaryRanges(vector<int>& nums) {
	vector<string> res;
	int n = nums.size();

	for (int i = 0; i < n; ++i) {
		int j = i;
		while (nums[i] + 1 == nums[i + 1]) ++i;
		if (i != j) res.push_back(to_string(nums[j]) + "->" + to_string(nums[i]));
		else res.push_back(to_string(nums[i]));
	}
	return res;
}




/* ========================== String Problems ============================ */
/* 415. Add Strings -- STRING */
/* Given two non-negative integers num1 and num2 represented as string,
* return the sum of num1 and num2. */
string addStrings(string num1, string num2) {
	string res("");
	int i = num1.size() - 1, j = num2.size() - 1, carry = 0;

	while (i >= 0 || j >= 0) {
		int a = i >= 0 ? num1[i--] - '0' : 0;
		int b = j >= 0 ? num2[j--] - '0' : 0;

		int sum = a + b + carry;
		carry = sum / 10;

		res = to_string(sum % 10) + res;
	}

	return carry == 1 ? "1" + res : res;
}

/* 38. Count and Say */
/* Given an integer n where 1 ≤ n ≤ 30, generate the nth term of the count-and-say sequence.
*  n = 5 =>  111221. */
string countAndSay(int n) {
	string res("1");
	while (--n) {
		string t("");
		int n = res.size();
		for (int i = 0; i < n; ++i) {
			int cnt = 1;
			while (i < n - 1 && res[i] == res[i + 1]) ++i, ++cnt;
			t += to_string(cnt) + res[i];
		}
		res = t;
	}
	return res;
}

/* 271. Encode and Decode Strings */
/* Design an algorithm to encode a list of strings to a string. The encoded string is then sent over 
* the network and is decoded back to the original list of strings. */
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

/* 171. Excel Sheet Column Number */
/* Given a column title as appear in an Excel sheet, return its corresponding column number. */
int titleToNumber(string s) {
	int res = 0;
	for (int i = 0; i < s.size(); ++i) {
		res = res * 26 + (s[i] - 'A') + 1;
	}
	return res;
}

/* 168. Excel Sheet Column Title */
/* Given a positive integer, return its corresponding column title as appear in an Excel sheet. */
string convertToTitle(int n) {
	return n <= 0 ? "" : convertToTitle(n / 26) + (char)('A' + --n % 26);
}

/* 28. Implement strStr() */
/* Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
* Input: haystack = "hello", needle = "ll". Output: 2 */
int strStr(string haystack, string needle) {
	int m = haystack.size(), n = needle.size();
	if (m < n) return -1;
	for (int i = 0; i <= m - n; ++i) {
		int j = 0;
		for (j = 0; j < n; ++j) {
			if (haystack[i + j] != needle[j]) break;
		}
		if (j == n) return i;
	}
	return -1;
}

/* 392. Is Subsequence */
/* Given a string s and a string t, check if s is subsequence of t. */
bool isSubsequence(string s, string t) {
	int m = s.size(), n = t.size(), j = 0;
	for (int i = 0; i < n; ++i) {
		if (t[i] == s[j]) ++j;
	}
	return j == m;
}

/* 9. Palindrome Number */
/* Determine whether an integer is a palindrome. An integer is a palindrome
* when it reads the same backward as forward. Input: 121. Output: true. */
bool isPalindrome(int x) {
	if (x < 0) return false;
	string s = to_string(x);
	int i = 0, j = s.size() - 1;
	while (i < j) {
		if (s[i++] != s[j--]) return false;
	}
	return true;
}

/* 1047. Remove All Adjacent Duplicates In String */
/* Given a string S of lowercase letters, a duplicate removal consists of
* choosing two adjacent and equal letters, and removing them.
* We repeatedly make duplicate removals on S until we no longer can.
* Return the final string after all such duplicate removals have been made.
* It is guaranteed the answer is unique. Input: "abbaca". Output: "ca" */
string removeDuplicates(string S) {
	string res("");
	for (auto c : S) {
		if (c == res.back()) {
			while (c == res.back()) res.pop_back();
		}
		else res += c;
	}
	return res;
}

/* 344. Reverse String */
/* Write a function that reverses a string. The input string is given as an array 
* of characters char[]. Do not allocate extra space for another array, you must
* do this by modifying the input array in-place with O(1) extra memory. 
* Input: ["h","e","l","l","o"]. Output: ["o","l","l","e","h"] */
void reverseString(vector<char>& s) {
	int i = 0, j = s.size() - 1;
	while (i < j) {
		swap(s[i++], s[j--]);
	}
}

/* 151. Reverse Words in a String */
/* Given an input string, reverse the string word by word.
* Input: "the sky is blue". Output: "blue is sky the". */
string reverseWords(string s) {
	if (s.empty()) return "";
	istringstream is(s);
	is >> s;
	string t;

	while (is >> t) s = t + " " + s;
	if (s.empty() || s[0] == '0') s = "";
	return s;
}

/* 214. Shortest Palindrome -- Two pointers & Recursion */
/* Given a string s, you are allowed to convert it to a palindrome by adding characters in front of it.
* Find and return the shortest palindrome you can find by performing this transformation.
* Input: "aacecaaa". Output: "aaacecaaa" */
string shortestPalindrome(string s) {
	int i = 0, n = s.size();
	for (int j = n - 1; j >= 0; --j) {
		if (s[i] == s[j]) ++i;
	}
	if (i == n) return s;
	string p = s.substr(i);
	reverse(p.begin(), p.end());
	return p + shortestPalindrome(s.substr(0, i)) + s.substr(i);
}

/* 443. String Compression */
/* Given an array of characters, compress it in-place.
* The length after compression must always be smaller than or equal to the original array.
* Every element of the array should be a character (not int) of length 1.
* After you are done modifying the input array in-place, return the new length of the array.
* Input: ["a","b","b","b","b","b","b","b","b","b","b","b","b"], Output: ["a","b","1","2"]. */
int compress(vector<char>& chars) {
	int n = chars.size(), res = 0, i = 0;

	while (i < n) {
		int j = i;
		while (j < n && chars[j] == chars[i]) ++j;
		chars[res++] = chars[i];

		if (j - i == 1) {
			i = j;
			continue;
		}
		// If "j - i == 12", then should add "1", "2" seperately. 
		for (auto c : to_string(j - i)) chars[res++] = c;
		i = j;
	}
	return res;
}




/* ========================== Math Problems ============================ */
/* 386. Lexicographical Numbers */
/* Given an integer n, return 1 - n in lexicographical order.
* For example, given 13, return: [1,10,11,12,13,2,3,4,5,6,7,8,9]. */
vector<int> lexicalOrder(int n) {
	vector<int> res(n, 0);
	int cur = 1;
	for (int i = 0; i < n; ++i) {
		res[i] = cur;
		if (cur * 10 <= n) {
			cur *= 10;
		}
		else {
			if (cur >= n) cur /= 10;
			++cur;
			while (cur % 10 == 0) cur /= 10;
		}
	}
	return res;
}


int main() {
	
	string s = "abc", p = "cba";
	cout << isAnagram(s, p) << endl; 
	
	/*
	int x = 8;
	cout << mySqrt(x) << endl; 
	/*
	string s("abee");
	cout << frequencySort(s) << endl; 
	/*
	vector<int> nums{ 1,3,-1,-3,5,3,6,7 };
	int k = 3; 
	vector<int> res = maxSlidingWindow(nums, k);
	printVector(res);
	
	// int target = 2; 
	// cout << searchRotatedArray(nums, target) << endl; 
	/*
	int n = 123; 
	cout << reverseInteger(n) << endl; 
	/*
	string s = "()())()";
	vector<string> res = removeInvalidParentheses(s);
	printVector(res);
	/*
	Interval a(0, 3);
	Interval b(5, 10);
	Interval c(15, 20);
	vector<Interval> intervals; 
	intervals.push_back(a);
	intervals.push_back(b);
	intervals.push_back(c);
	cout << minMeetingRooms(intervals) << endl;
	/*
	string s = "abcabcdbb";
	cout << lengthOfLongestSubstring2(s) << endl; 
	//cout << longestPalindrome(s) << endl; 
	/*
	vector<int> nums{ 100, 4, 200, 1, 3, 2, 0, 6 };
	cout << longestConsecutive(nums) << endl; 
	/*
	int n = 968;
	cout << intToRoman(n) << endl;
	*/

	/*
	RandomizedSet* obj = new RandomizedSet(); 
	obj->insert(1);
	obj->insert(2); 
	obj->insert(3);
	obj->insert(4);
	obj->insert(5);
	obj->getRandom();
	obj->printSet(); 
	/*
	int n = 81; 
	cout << isHappy(n) << endl; 
	/*
	int n = 3; 
	vector<string> res = generateParenthesis(n);
	printVector(res);
	/*
	string s("leetcodelover");
	cout << firstUniqChar(s) << endl; 
	/*
	HitCounter* obj = new HitCounter();
	int timestamp = 1; 
	obj -> hit(timestamp);
	int param_2 = obj->getHits(timestamp);
	cout << param_2 << endl; 
	/*
	string s("3[a1[bc2[e]]]");
	cout << decodeString(s) << endl; 
	*/
	
	//vector<int> vec{3, 21, 2, 5, 6, 7, 9};
	//cout << findPeakElement(vec) << endl; 

	/*
	vector<int> res = dailyTemperatures(vec);
	printVector(res);
	*/
}

