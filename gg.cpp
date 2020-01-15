#include<iostream>
#include<algorithm>
#include<numeric>
#include<vector>
#include<stack>
#include<string>
#include<queue>
#include<sstream>
#include<deque>
#include <functional> 
#include<set>
#include<map>
#include<unordered_map>
#include<unordered_set>
#include"gg.h"
#include<bitset>
using namespace std;


// ===========================================================
vector<vector<int>> dirs{ { -1, 0 },{ 1, 0 },{ 0, -1 },{ 0, 1 } };
vector<vector<int>> dirs2{ { -2, -1 },{ -2, 1 },{ -1, 2 },{ 1, 2 },{ 2, 1 },
{ 2, -1 },{ 1, -2 },{ -1, -2 } };
vector<vector<int>> dirs3{ { -1, -1 },{ -1, 0 },{ -1, 1 },{ 0, -1 },{ 0, 1 },
{ 1, -1 },{ 1, 0 },{ 1, 1 } };

void printGrid(vector<vector<int>>& grid) {
	for (int i = 0; i < grid.size(); ++i) {
		for (int j = 0; j < grid[i].size(); ++j) {
			cout << grid[i][j] << " ";
		}
		cout << endl;
	}
}
// ===========================================================


// =================== 1. DP PROBLEMS ========================
/* 70. Climbing Stairs */
/* You are climbing a stair case. It takes n steps to reach to the top.
* Each time you can either climb 1 or 2 steps. In how many
* distinct ways can you climb to the top? */
int climbStairs(int n) {
	vector<int> dp(n + 1, 0);
	dp[0] = 1, dp[1] = 1;
	for (int i = 2; i <= n; ++i) {
		dp[i] = dp[i - 1] + dp[i - 2];
	}
	return dp.back();
}

/* 322. Coin change */
/* You are given coins of different denominations and a total amount
* of money amount. Write a function to compute the FEWEST number
* of coins that you need to make up that amount. If that amount
* of money cannot be made up by any combination of the coins, return -1.
* Input: coins = [1, 2, 5], amount = 11. Output: 3. Explanation: 11 = 5 + 5 + 1. */
int coinChange(vector<int>& coins, int amount) {
	vector<int> dp(amount + 1, amount + 1);
	dp[0] = 0;
	for (int i = 1; i <= amount; ++i) {
		for (auto a : coins) {
			if (i >= a) dp[i] = min(dp[i], 1 + dp[i - a]);
		}
	}
	// IMPORTANT.
	return dp.back() == amount + 1 ? -1 : dp.back();
}

/* 518. Coin Change 2 */
/* You are given coins of different denominations and a total amount of money. 
* Write a function to compute the number of combinations that make up that amount.
* You may assume that you have infinite number of each kind of coin. 
* Input: amount = 5, coins = [1, 2, 5]. Output: 4. 
* 5=5, 5=2+2+1, 5=2+1+1+1, 5=1+1+1+1+1.
* dp[i][j] 表示用前i个硬币组成钱数为j的不同组合方法，怎么算才不会重复，也不会漏掉呢？
* 我们采用的方法是一个硬币一个硬币的增加，每增加一个硬币，都从1遍历到 amount，
* 对于遍历到的当前钱数j，组成方法就是不加上当前硬币的拼法 dp[i-1][j]，
* 还要加上，去掉当前硬币值的钱数的组成方法. */
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

/* 377. Combination Sum IV -- COMPARE WITH "coin change 2" */
/* Given an integer array with all positive numbers and
* no duplicates, find the number of possible COMBINATIONS
* that add up to a positive integer target.
* Example: nums = [1, 2, 3], target = 4. Output is 7. 
* (1, 1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 3), (2, 1, 1), (2, 2), (3, 1). 
* dp[i] 表示目标数为i的解的个数，然后从1遍历到 target，对于每一个数i，遍历 nums 数组，
* 如果 i>=x, dp[i] += dp[i - x]。这个也很好理解，比如说对于 [1,2,3], 4，这个例子，
* 当计算 dp[3] 的时候，3可以拆分为 1+x，而x即为 dp[2]，3也可以拆分为 2+x，此时x为 dp[1]，
* 3同样可以拆为 3+x，此时x为 dp[0]，把所有的情况加起来就是组成3的所有情况了. */
int combinationSum4(vector<int>& nums, int target) {
	vector<int> dp(target + 1, 0);
	dp[0] = 1;
	for (int i = 1; i <= target; ++i) {
		for (auto a : nums) {
			if (i >= a) dp[i] += dp[i - a];
		}
	}
	return dp.back();
}

/* 651. 4 Keys Keyboard */
/* You can only press the keyboard for N times (with the above four keys), find out 
 * the maximum numbers of 'A' you can print on screen. Key 1: (A): Print one 'A' on screen.
 * Key 2: (Ctrl-A): Select the whole screen.. Key 3: (Ctrl-C): Copy selection to buffer.
 * Key 4: (Ctrl-V): Print buffer on screen appending it after what has already been printed.*/
int maxA(int N) {
	int res = N;
	for (int i = 1; i < N - 2; ++i) {
		res = max(res, maxA(i) * (N - 1 - i));
	}
	return res;
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

// 351. Android Unlock Patterns
 int numberOfPatterns(int m, int n, vector<vector<int>>& jumps, vector<int>& visited, int num, int res, int len) {
	if (len >= m) ++res;
	++len;
	if (len > n) return res;
	visited[num] = 1;

	for (int i = 1; i <= 9; ++i) {
		int jump = jumps[num][i];
		if (!visited[i] && (jump == 0 || visited[jump])) {
			res = numberOfPatterns(m, n, jumps, visited, i, res, len);
		}
	}

	visited[num] = 0; // BACKTRACKING
	return res;
}

int numberOfPatterns(int m, int n) {
	int res = 0;
	vector<vector<int>> jumps(10, vector<int>(10, 0));
	jumps[1][3] = jumps[3][1] = 2;
	jumps[4][6] = jumps[6][4] = 5;
	jumps[7][9] = jumps[9][7] = 8;
	jumps[1][7] = jumps[7][1] = 4;
	jumps[2][8] = jumps[8][2] = 5;
	jumps[3][9] = jumps[9][3] = 6;
	jumps[1][9] = jumps[9][1] = 5;
	jumps[3][7] = jumps[7][3] = 5;

	vector<int> visited(10, 0);

	res += numberOfPatterns(m, n, jumps, visited, 1, 0, 1) * 4;
	res += numberOfPatterns(m, n, jumps, visited, 2, 0, 1) * 4;
	res += numberOfPatterns(m, n, jumps, visited, 5, 0, 1);

	return res;
}

/* 361. Bomb Enemy */
/* Given a 2D grid, each cell is either a wall 'W',
* an enemy 'E' or empty '0' (the number zero),
* return the maximum enemies you can kill using ONE bomb.
* The bomb kills all the enemies in the same row and column
* from the planted point until it hits the wall since the
* wall is too strong to be destroyed.
* Note: You can only put the bomb at an empty cell. */
int maxKilledEnemies(vector<vector<char>>& grid) {
	int m = grid.size(), n = grid[0].size(), res = 0;
	vector<vector<int>> v1(m, vector<int>(n, 0)), v2(v1), v3(v1), v4(v1);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			auto t = (j == 0 || grid[i][j] == 'W') ? 0 : v1[i][j - 1];
			v1[i][j] = grid[i][j] == 'E' ? t + 1 : t;
		}
		for (int j = n - 1; j >= 0; --j) {
			auto t = (j == n - 1 || grid[i][j] == 'W') ? 0 : v2[i][j + 1];
			v2[i][j] = grid[i][j] == 'E' ? t + 1 : t;
		}
	}
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < m; ++i) {
			auto t = (i == 0 || grid[i][j] == 'W') ? 0 : v3[i - 1][j];
			v3[i][j] = grid[i][j] == 'E' ? t + 1 : t;
		}
		for (int i = m - 1; i >= 0; --i) {
			auto t = (i == m - 1 || grid[i][j] == 'W') ? 0 : v4[i + 1][j];
			v4[i][j] = grid[i][j] == 'E' ? t + 1 : t;
		}
	}

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == '0')
				res = max(res, v1[i][j] + v2[i][j] + v3[i][j] + v4[i][j]);
		}
	}
	return res;
}

/* 312. Burst Balloons -- HARD */
/* Given n balloons, indexed from 0 to n-1. Each balloon is painted with
* a number on it represented by array nums. You are asked to burst all
* the balloons. If the you burst balloon i you will get
* nums[left] * nums[i] * nums[right] coins. Here left and right are
* adjacent indices of i. After the burst, the left and right then
* becomes adjacent. Find the maximum coins you can collect by bursting
* the balloons wisely. Input: [3,1,5,8], Output: 167. */
int maxCoins(vector<int>& nums) {
	int n = nums.size();
	nums.insert(nums.begin(), 1);
	nums.push_back(1);
	vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
	// 需要先更新完所有的小区间，然后才能去更新大区间
	for (int len = 1; len <= n; ++len) {
		for (int i = 1; i <= n - len + 1; ++i) {
			int j = i + len - 1;
			for (int k = i; k <= j; ++k) {
				dp[i][j] = max(dp[i][j], dp[i][k - 1] + dp[k + 1][j] + nums[i - 1] * nums[k] * nums[j + 1]);
			}
		}
	}
	return dp[1][n];
}

/* 464. Can I Win */
/* In the "100 game," two players take turns adding, to a running total,
* any integer from 1..10. The player who first causes the running total
* to reach or exceed 100 wins. Players cannot re-use integers.
* Determine if the first player to move can force a win, assuming both
* players play optimally.
* 使用一个整型数按位来记录数组中的某个数字是否使用过，我们遍历所有数字，
* 将该数字对应的 mask 算出来，如果其和 used 相与为0的话，说明该数字没有使用过. */
bool canIWin(int len, int total, int used, unordered_map<int, int>& m) {
	if (m.count(used)) return m[used];

	for (int i = 0; i < len; ++i) {
		int cur = (1 << i);
		if ((used & cur) == 0) {
			if (total <= i + 1 || !canIWin(len, total - (i + 1), (cur | used), m)) {
				return m[used] = 1;
			}
		}
	}
	return m[used] = 0;
}

bool canIWin(int maxChoosableInteger, int desiredTotal) {
	if (maxChoosableInteger >= desiredTotal) return true;
	if (maxChoosableInteger * (maxChoosableInteger + 1) / 2 < desiredTotal) return false;
	unordered_map<int, int> m;
	return canIWin(maxChoosableInteger, desiredTotal, 0, m);
}

/* 486. Predict the Winner */
/* Given an array of scores that are non-negative integers.
* Player 1 picks one of the numbers from either end of the array
* followed by the player 2 and then player 1 and so on. Each time
* a player picks a number, that number will not be available for
* the next player. This continues until all the scores have been
* chosen. The player with the maximum score wins. Given an array
* of scores, predict whether player 1 is the winner. You can
* assume each player plays to maximize his score. */
/* Solution 1: Brute force.  */
bool PredictTheWinner(vector<int> nums, int sum1, int sum2, int player) {
	if (nums.empty()) return sum1 >= sum2;
	if (nums.size() == 1) {
		if (player == 1) return sum1 + nums[0] >= sum2;
		else if (player == 2) return sum2 + nums[0] > sum1;
	}
	vector<int> va = vector<int>(nums.begin() + 1, nums.end());
	vector<int> vb = vector<int>(nums.begin(), nums.end() - 1);
	if (player == 1) {
		return !PredictTheWinner(va, sum1 + nums[0], sum2, 2) || 
			!PredictTheWinner(vb, sum1 + nums.back(), sum2, 2);
	}
	else if (player == 2) {
		return !PredictTheWinner(va, sum1, sum2 + nums[0], 1) || 
			!PredictTheWinner(vb, sum1, sum2 + nums.back(), 1);
	}
}

bool PredictTheWinner(vector<int>& nums) {
	return PredictTheWinner(nums, 0, 0, 1);
}

/* Solution 2: dp[i][j] saves how much MORE scores that the first-in-action player
* will get from i to j THAN the second player.
* TOP DOWN DP, DP is used to save inter step results. */
int PredictTheWinner2(vector<int>& nums, int i, int j, vector<vector<int> >& dp) {
	if (dp[i][j] == -1) {
		/* (1) Second player can get "PredictTheWinner(nums, i + 1, j, dp)" if first get "nums[i]"
		* (2) Second player can get "PredictTheWinner(nums, i, j - 1, dp)" if first get "nums[j]"*/
		dp[i][j] = (i == j) ? nums[i] : max(nums[i] - PredictTheWinner2(nums, i + 1, j, dp),
			nums[j] - PredictTheWinner2(nums, i, j - 1, dp));
	}
	return dp[i][j];
}

bool PredictTheWinner2(vector<int>& nums) {
	int n = nums.size();
	vector<vector<int> > dp(n, vector<int>(n, -1));
	return PredictTheWinner2(nums, 0, n - 1, dp) >= 0;
}

/* 799. Champagne Tower */
/* Now after pouring some non-negative integer cups of champagne, return how full 
* the j-th glass in the i-th row is (both i and j are 0 indexed.) */
double champagneTower(int poured, int query_row, int query_glass) {
	int n = 100;
	vector<vector<double> > dp(n + 1, vector<double>(n + 1, 0.0));
	dp[0][0] = poured;

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j <= i; ++j) {
			if (dp[i][j] >= 1) {
				dp[i + 1][j] += (dp[i][j] - 1) / 2;
				dp[i + 1][j + 1] += (dp[i][j] - 1) / 2;
				dp[i][j] = 1.0;
			}
		}
	}
	return dp[query_row][query_glass];
}

/* 656. Coin Path -- CLASSIC DP PROBLEM -- HARD */
/* Given an array A (index starts at 1) consisting of N integers: A1, A2, ..., AN and an integer B.
* The integer B denotes that from any place (suppose the index is i) in the array A, you can
* jump to any one of the place in the array A indexed i+1, i+2, …, i+B if this place can be jumped to.
* you start from the place indexed 1 in the array A, and your aim is to reach the place indexed N
* using the minimum coins. You need to return the path of indexes
*  Input: [1,2,4,-1,2], 2. Output: [1,3,5]. */
vector<int> cheapestJump(vector<int>& A, int B) {
	if (A.back() == -1) return {};
	int n = A.size();
	vector<int> res, dp(n, INT_MAX), pos(n, -1);
	dp[n - 1] = A[n - 1];

	for (int i = n - 2; i >= 0; --i) {
		if (A[i] == -1) continue;
		for (int j = i + 1; j <= min(i + B, n - 1); ++j) {
			if (dp[j] == INT_MAX) continue;
			if (A[i] + dp[j] < dp[i]) {
				dp[i] = A[i] + dp[j];
				pos[i] = j;
			}
		}
	}

	if (dp[0] == INT_MAX) return res;

	for (int i = 0; i != -1; i = pos[i]) {
		res.push_back(i + 1);
	}
	return res;
}

/* 357. Count Numbers with Unique Digits */
/* Given a non-negative integer n, count all numbers with
* unique digits, x, where 0 ≤ x < 10^n.
* Example: Input: 2. Output: 91. */
int countNumberHelper(int n) {
	if (n <= 0) return 0;
	if (n == 1) return 10;
	int res = 1;
	for (int i = 9; i >= 9 - n + 2; --i) {
		res *= i;
	}
	return res * 9;
}

int countNumbersWithUniqueDigits(int n) {
	int res = 0;
	if (n == 0) return 0;
	for (int i = 1; i <= n; ++i) {
		res += countNumberHelper(i);
	}
	return res;
}

/* 91. Decode ways */
/* A message containing letters from A-Z is being encoded to
* numbers using the following mapping:
* 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26
* Given a non-empty string containing only digits, determine
* the total number of ways to decode it.
* Input: "226". Output: 3. */
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

/* 583. Delete Operation for Two Strings */
/* Given two words word1 and word2, find the minimum number of steps required to make 
* word1 and word2 the same, where in each step you can delete one character in either string. 
* Input: "sea", "eat". Output: 2. 
* Logic: the same as finding longest common substring. */
int minDistance(string word1, string word2) {
	int m = word1.size(), n = word2.size();
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (word1[i - 1] == word2[j - 1]) {
				dp[i][j] = 1 + dp[i - 1][j - 1];
			}
			else {
				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
			}
		}
	}
	return m + n - 2 * dp.back().back();
}

/* 115. Distinct Subsequences */
/* Given a string S and a string T, count the number of distinct
* subsequences of S which EQUALS T. A subsequence of a string is
* a new string which is formed from the original string by deleting
* some (can be none) of the characters without disturbing the
* relative positions of the remaining characters.
* Input: S = "rabbbit", T = "rabbit". Output: 3. */
int numDistinct(string s, string t) {
	int m = s.size(), n = t.size();
	if (m < n) return 0;
	vector<vector<long long>> dp(m + 1, vector<long long>(n + 1, 0));

	for (int i = 0; i <= m; ++i) dp[i][0] = 1;
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (s[i - 1] == t[j - 1]) dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
			else dp[i][j] = dp[i - 1][j];
		}
	}
	return (int)dp.back().back();
}

/* 940. Distinct Subsequences II -- HARD */
/* Given a string S, count the number of distinct, non-empty
* subsequences of S. Since the result may be large,
* return the answer modulo 10^9 + 7.
* Input: "abc". Output: 7. Explanation: The 7 distinct
* subsequences are "a", "b", "c", "ab", "ac", "bc", and "abc".
* Input: "aba". Output: 6. Explanation: The 6 distinct
* subsequences are "a", "b", "ab", "ba", "aa" and "aba".
* LOGIC: Init an array endswith[26] endswith[i] to count
* how many sub sequence that ends with ith character.
* Now we have N = sum(endswith) different sub sequence,
* add a new character c to each of them, then we have N
* different sub sequence that ends with c. With this idea,
* we loop on the whole string S, and we update end[c] = sum(end) + 1
* for each character. We need to plus one here, because "c"
* itself is also a sub sequence.*/
int distinctSubseqII(string S) {
	long endWith[26] = {}, mod = 1e9 + 7;
	for (auto c : S) {
		endWith[c - 'a'] = accumulate(begin(endWith), end(endWith), 1L) % mod;
	}
	return accumulate(begin(endWith), end(endWith), 0L) % mod;
}

/* 72. Edit Distance */
/* Given two words word1 and word2, find the MINIMUM number of
* operations required to convert word1 to word2.
* You have the following 3 operations permitted on a word:
* Insert a character, Delete a character, Replace a character
* Example 1: Input: word1 = "horse", word2 = "ros". Output: 3. */
int minDistance(string word1, string word2) {
	int m = word1.size(), n = word2.size();
	vector<vector<int> > dp(m + 1, vector<int>(n + 1, 0));
	for (int i = 0; i <= m; ++i) dp[i][0] = i;
	for (int j = 0; j <= n; ++j) dp[0][j] = j;

	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (word1[i - 1] == word2[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1];
			}
			else {
				dp[i][j] = 1 + min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1]));
			}
		}
	}
	return dp.back().back();
}

/* 790. Domino and Tromino Tiling */
/* We have two types of tiles: a 2x1 domino shape, and
* an "L" tromino shape. These shapes may be rotated.
* XX  <- domino, XX  <- "L" tromino
*                X
* Given N, how many ways are there to tile a 2 x N board?
* Return your answer modulo 10^9 + 7. */
int numTilings(int n) {
	long long mod = 1e9 + 7;
	vector<long long> dp(n + 1, 0);
	dp[0] = 1, dp[1] = 1, dp[2] = 2;
	for (int i = 3; i <= n; ++i) {
		dp[i] = (2 * dp[i - 1] + dp[i - 3]) % mod;
	}
	return dp.back();
}

/* 1105. Filling Bookcase Shelves -- CLASSIC DP PROBLEM */
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

/* 926. Flip String to Monotone Increasing */
/* A string of '0's and '1's is monotone increasing if it consists of some number of
* '0's (possibly 0), followed by some number of '1's (also possibly 0.) We are given
* a string S of '0's and '1's, and we may flip any '0' to a '1' or a '1' to a '0'.
* Return the minimum number of flips to make S monotone increasing.
* Input: "00110". Output: 1. Input: "010110". Output: 2. Input: "00011000". Output: 2. */
int minFlipsMonoIncr(string S) {
	int n = S.size(), res = INT_MAX;
	vector<int> dp1(n + 1, 0), dp2(n + 1, 0);

	for (int i = 1, j = n - 1; i <= n, j >= 0; ++i, --j) {
		dp1[i] += dp1[i - 1] + (S[i - 1] == '0' ? 0 : 1);
		dp2[j] += dp2[j + 1] + (S[j] == '1' ? 0 : 1);
	}
	for (int i = 0; i <= n; ++i) {
		res = min(res, dp1[i] + dp2[i]);
	}
	return res;
}

/* 514. Freedom Trail -- DP PROBLEM */
/* Given a string ring, which represents the code engraved on
* the outer ring and another string key, which represents
* the keyword needs to be spelled. You need to find the
* MINIMUM number of steps in order to spell all the
* characters in the keyword.
* At the stage of rotating the ring to spell the key character key[i]:
* (1) You can rotate the ring clockwise or anticlockwise one place,
*     which counts as 1 step. The final purpose of the rotation
*     is to align one of the string ring's characters at the 12:00 direction,
*     where this character must equal to the character key[i].
* (2) If the character key[i] has been aligned at the 12:00 direction,
* you need to press the center button to spell, which also counts as 1 step.
* After the pressing, you could begin to spell the next character
* in the key (next stage), otherwise, you've finished all the spelling. */
// DEFINE: dp[i][j] := the minimum steps to make ring[j] match with key[i].
int findRotateSteps(string ring, string key) {
	int m = key.size(), n = ring.size();
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

	for (int i = m - 1; i >= 0; --i) {
		for (int j = 0; j < n; ++j) {
			dp[i][j] = INT_MAX; // IMPORTANT. Avoid the stack over flow

			for (int k = 0; k < n; ++k) {
				if (ring[k] == key[i]) {
					int diff = abs(j - k);
					int step = min(diff, n - diff);
					dp[i][j] = min(dp[i][j], step + dp[i + 1][k]);
				}
			}
		}
	}
	return dp[0][0] + m;
}

/* 403. Frog Jump */
/* Given a list of stones' positions (in units) in sorted ascending order,
* determine if the frog is able to cross the river by landing on the last
* stone. Initially, the frog is on the first stone and assume the first
* jump must be 1 unit. If the frog's last jump was k units, then its next
* jump must be either k - 1, k, or k + 1 units. Note that the frog can
* only jump in the forward direction.
* [0,1,3,5,6,8,12,17] */
bool canCross(vector<int>& stones, int ix, int jump, unordered_map<int, int>& m) {
	int key = (ix << 10) | jump, n = stones.size();
	if (ix >= n - 1) return true;
	if (m.count(key)) return m[key];

	for (int i = ix + 1; i < stones.size(); ++i) {
		int dist = stones[i] - stones[ix];
		if (dist < jump - 1) continue;
		if (dist > jump + 1) return m[key] = false;
		if (canCross(stones, i, dist, m)) return m[key] = true;
	}
	return m[key];
}

bool canCross(vector<int>& stones) {
	unordered_map<int, int> m;
	return canCross(stones, 0, 0, m);
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

/* 688. Knight Probability in Chessboard */
/* On an NxN chessboard, a knight starts at the r-th row and c-th
* column and attempts to make exactly K moves. The rows and
* columns are 0 indexed, so the top-left square is (0, 0), and
* the bottom-right square is (N-1, N-1). A chess knight has 8
* possible moves it can make, as illustrated below. Return the
* probability that the knight remains on the board after it
* has stopped moving.*/
double knightProbability(int N, int K, int r, int c) {
	if (K == 0) return 1;
	vector<vector<double> > dp(N, vector<double>(N, 1));
	for (int k = 0; k < K; ++k) {
		vector<vector<double>> t(N, vector<double>(N, 0));

		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				for (auto dir : dirs2) {
					int x = i + dir[0], y = j + dir[1];
					if (x < 0 || x >= N || y < 0 || y >= N) continue;
					t[i][j] += dp[x][y];
				}
			}
		}
		dp = t;
	}
	return dp[r][c] / pow(8, K);
}

/* 368. Largest Divisible Subset */
/* Given a set of distinct positive integers, find the largest
* subset such that every pair (Si, Sj) of elements in this
* subset satisfies: Si % Sj = 0 or Sj % Si = 0.
* If there are multiple solutions, return any subset is fine.
* Input: [1,2,4,8]. Output: [1,2,4,8]. */
vector<int> largestDivisibleSubset(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	int n = nums.size(), len = 0, ix = 0;
	vector<int> dp(n, 0), parent(n, 0), res;
	for (int i = n - 1; i >= 0; --i) {
		for (int j = i; j < n; ++j) {
			if (nums[j] % nums[i] == 0 && dp[i] < dp[j] + 1) {
				dp[i] = dp[j] + 1;
				parent[i] = j;
			}
			if (len < dp[i]) {
				len = dp[i];
				ix = i;
			}
		}
	}
	for (int i = 0; i < len; ++i) {
		res.push_back(nums[ix]);
		ix = parent[ix];
	}
	return res;
}

/* 764. Largest Plus Sign */
/* In a 2D grid from (0, 0) to (N-1, N-1), every cell contains a 1,
* except those cells in the given list mines which are 0.
* What is the largest axis-aligned plus sign of 1s contained
* in the grid? Return the order of the plus sign. If there is none,
* return 0. */
int orderOfLargestPlusSign(int N, vector<vector<int>>& mines) {
	int res = 0;
	vector<vector<int>> dp(N, vector<int>(N, N));
	for (auto m : mines) {
		dp[m[0]][m[1]] = 0;
	}
	for (int i = 0; i < N; ++i) {
		for (int j = 0, k = N - 1, l = 0, r = 0, u = 0, d = 0; j < N; ++j, --k) {
			dp[i][j] = min(dp[i][j], l = (dp[i][j] == 0 ? 0 : l + 1));
			dp[i][k] = min(dp[i][k], r = (dp[i][k] == 0 ? 0 : r + 1));
			dp[j][i] = min(dp[j][i], u = (dp[j][i] == 0 ? 0 : u + 1));
			dp[k][i] = min(dp[k][i], d = (dp[k][i] == 0 ? 0 : d + 1));
		}
	}
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			res = max(res, dp[i][j]);
		}
	}
	return res;
}

/* 813. Largest Sum of Averages */
/* We partition a row of numbers A into at most K adjacent (non-empty) groups, 
* then our score is the sum of the average of each group. What is the largest score we can achieve? 
* Input: A = [9,1,2,3,9], K = 3. Output: 20.
* dp[i][k]表示范围是[i, n-1]的子数组分成k组的最大得分问题就转换成了从k-1组变成k组，即多分出一组，
* 那么在范围[i, n-1]多分出一组，实际上就是将其分成两部分，一部分是一组，另一部分是k-1组. */
double largestSumOfAverages(vector<int>& A, int K) {
	int n = A.size();
	vector<vector<double>> dp(n, vector<double>(K));
	vector<double> sums(n + 1, 0);
	for (int i = 0; i < n; ++i) {
		sums[i + 1] = sums[i] + A[i];
	}
	for (int i = 0; i < n; ++i) {
		dp[i][0] = (sums[n] - sums[i]) / (n - i);
	}
	for (int k = 1; k < K; ++k) {
		for (int i = 0; i < n - 1; ++i) {
			for (int j = i + 1; j < n; ++j) {
				dp[i][k] = max(dp[i][k], dp[j][k - 1] + (sums[j] - sums[i]) / (j - i));
			}
		}
	}
	return dp[0].back();
}

/* 413. Arithmetic Slices */
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

/* 446. Arithmetic Slices II - Subsequence */
/* A sequence of numbers is called arithmetic if it consists
* of at least three elements and if the difference between
* ANY two consecutive elements is the same.
* Input: [2, 4, 6, 8, 10]. Output: 7
* Explanation: All arithmetic subsequence slices are:
* [2,4,6], [4,6,8], [6,8,10], [2,4,6,8], [4,6,8,10],
* [2,4,6,8,10], [2,6,10]. */
int numberOfArithmeticSlices(vector<int>& A) {
	int res = 0, n = A.size();
	vector<unordered_map<int, int>> m(n);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < i; ++j) {
			long long diff = (long long)A[i] - A[j];
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

/* 1027. Longest Arithmetic Sequence */
/* Given an array A of integers, return the length of the longest arithmetic subsequence in A.
 * Input: [9,4,7,2,10]. Output: 3. Explanation: The longest arithmetic subsequence is [4,7,10]. */
int longestArithSeqLength(vector<int>& A) {
	int n = A.size(), res = 2; // Initizalization
	unordered_map<int, unordered_map<int, int>> m; 

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < i; ++j) {
			int d = A[i] - A[j]; 
			m[d][i] = m[d].count(j) ? m[d][j] + 1 : 2; 
			res = max(res, m[d][i]);
		}
	}
	return res;
}

/* 1218. Longest Arithmetic Subsequence of Given Difference */
/* Given an integer array arr and an integer difference, return the length of the 
* longest subsequence in arr which is an arithmetic sequence such that the difference
* between adjacent elements in the subsequence equals difference. 
* Input: arr = [1,5,7,8,5,3,4,2,1], difference = -2. Output: 4. */
int longestSubsequence(vector<int>& arr, int difference) {
	int n = arr.size(), res = 0;
	unordered_map<int, int> m; // {num, length} mapping
	for (auto a : arr) {
		res = max(res, m[a] = 1 + m[a - difference]);
	}
	return res; 
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

/* 329. Longest Increasing Path in a Matrix */
/* Given an integer matrix, find the length of the longest increasing path.
* From each cell, you can either move to four directions:
* left, right, up or down. You may NOT move diagonally or
* move outside of the boundary (i.e. wrap-around is not allowed).
* Input: nums =
[[9,9,4],
[6,6,8],
[2,1,1]] . Output: 4 */
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

/* 300. Longest Increasing Subsequence */
/* Given an unsorted array of integers, find the length
* of longest increasing subsequence.
* Example: Input: [10,9,2,5,3,7,101,18]. Output: 4.
* Follow up: Could you improve it to O(n log n) time complexity? */
int lengthOfLIS(vector<int>& nums) {
	int n = nums.size(), res = 0;
	vector<int> dp(n, 1);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < i; ++j) {
			if (nums[j] < nums[i]) dp[i] = max(dp[i], 1 + dp[j]);
		}
		res = max(res, dp[i]);
	}
	return res;
}

/* 5. Longest Palindromic Substring */
/* Given a string s, find the longest palindromic substring in s.
* You may assume that the maximum length of s is 1000.*/
string longestPalindrome(string s) {
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

/* 363. Max Sum of Rectangle No Larger Than K */
/* Given a non-empty 2D matrix matrix and an integer k, find the max sum
* of a rectangle in the matrix such that its sum is no larger than k. */
int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
	int m = matrix.size(), n = matrix[0].size(), res = INT_MIN;
	vector<vector<int>> sums(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			int t = matrix[i][j];
			if (i > 0) t += sums[i - 1][j];
			if (j > 0) t += sums[i][j - 1];
			if (i > 0 && j > 0) t -= sums[i - 1][j - 1];

			sums[i][j] = t;
			for (int r = 0; r <= i; ++r) {
				for (int c = 0; c <= j; ++c) {
					int d = sums[i][j];

					if (r > 0) d -= sums[r - 1][j];
					if (c > 0) d -= sums[i][c - 1];
					if (r > 0 && c > 0) d += sums[r - 1][c - 1];
					if (d <= k) res = max(res, d);
				}
			}
		}
	}
	return res;
}

/* 85. Maximal Rectangle */
/* Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle
* containing only 1's and return its area. Example: Input:
* [["1","0","1","0","0"],
   ["1","0","1","1","1"],
   ["1","1","1","1","1"],
   ["1","0","0","1","0"]]. Output: 6.  */
int histogram(vector<int>& height) {
	height.push_back(0); 
	int n = height.size(), res = 0; 
	stack<int> st; 

	for (int i = 0; i < n; ++i) {
		while (!st.empty() && height[i] <= height[st.top()]) {
			auto t = st.top(); st.pop(); 
			// IMPORTANT: "height[t]"
			res = max(res, height[t] * (st.empty() ? i : i - st.top() - 1));
		}
		st.push(i);
	}
	return res; 
}

int maximalRectangle(vector<vector<char>>& matrix) {
	if (matrix.empty() || matrix[0].empty()) return 0; 
	int m = matrix.size(), n = matrix[0].size(), res = 0; 
	vector<int> height(n, 0);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			height[j] = matrix[i][j] == '0' ? 0 : 1 + height[j]; 
		}
		res = max(res, histogram(height));
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

/* 152. Maximum Product Subarray */
/* Given an integer array nums, find the contiguous subarray
* within an array (containing at least one number) which has
* the largest product. Input: [2,3,-2,4]. Output: 6. */
int maxProduct(vector<int>& nums) {
	if (nums.empty()) return 0;
	int n = nums.size(), res = nums[0];
	vector<int> mx(n, 0), mn(n, 0);
	mn[0] = nums[0], mx[0] = nums[0];

	for (int i = 1; i < n; ++i) {
		mn[i] = min(nums[i], min(mn[i - 1] * nums[i], mx[i - 1] * nums[i]));
		mx[i] = max(nums[i], max(mn[i - 1] * nums[i], mx[i - 1] * nums[i]));
		res = max(res, mx[i]);
	}
	return res;
}

/* 568. Maximum Vacation Days */
/* ou can only travel among N cities, represented by indexes from 0 to N-1.
* Initially, you are in the city indexed 0 on Monday. N*K matrix called days
* representing this relationship. For the value of days[i][j], it represents
* the maximum days you could take vacation in the city i in the week j. */
int maxVacationDays(vector<vector<int>>& flights, vector<vector<int>>& days) {
	int m = flights.size(), n = days[0].size(), res = 0;
	vector<vector<int> > dp(m, vector<int>(n, 0));

	for (int j = n - 1; j >= 0; --j) {
		for (int i = 0; i < m; ++i) {
			dp[i][j] = days[i][j];

			for (int k = 0; k < m; ++k) {
				if (j < n - 1 && (flights[i][k] || i == k)) {
					dp[i][j] = max(dp[i][j], dp[k][j + 1] + days[i][j]);
				}
			}
			if ((i == 0 || flights[0][i]) && j == 0) res = max(res, dp[i][j]);
		}
	}
	return res;
}

/* 931. Minimum Falling Path Sum */
/* Given a square array of integers A, we want the minimum sum of a falling
* path through A. A falling path starts at any element in the first row,
* and chooses one element from each row.  The next row's choice must be in
* a column that is different from the previous row's column by at most one. */
int minFallingPathSum(vector<vector<int>>& A) {
	int n = A.size(); 
	for (int i = 1; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			A[i][j] += min(A[i - 1][j], min(A[i - 1][max(0, j - 1)], A[i - 1][min(j + 1, n - 1)]));
		}
	}
	return *min_element(A[n - 1].begin(), A[n - 1].end());
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

/* 801. Minimum Swaps To Make Sequences Increasing */
/* We have two integer sequences A and B of the same non-zero length.
* We are allowed to swap elements A[i] and B[i].  Note that both elements
* are in the same index position in their respective sequences.
* At the end of some number of swaps, A and B are both strictly
* increasing. Given A and B, return the minimum number of swaps to make
* both sequences strictly increasing.  It is guaranteed that the given
* input always makes it possible.
* Input: A = [1,3,5,4], B = [1,2,3,7]. Output: 1.
* swap[i]表示范围[0, i]的子数组同时严格递增且当前位置i需要交换的最小交换次数.
* noSwap[i]表示范围[0, i]的子数组同时严格递增且当前位置i不交换的最小交换次数. */
int minSwap(vector<int>& A, vector<int>& B) {
	int n = A.size(); 
	vector<int> swap(n, n), keep(n, n);
	swap[0] = 1, keep[0] = 0; // IMPORTANT initizalization

	for (int i = 1; i < n; ++i) {
		// if keep
		if (A[i - 1] < A[i] && B[i - 1] < B[i]) {
			keep[i] = keep[i - 1]; 
			swap[i] = swap[i - 1] + 1; // 如果当前位置要交换， 那么前一个位置都要交换。
		}
		// if swap
		if (A[i - 1] < B[i] && B[i - 1] < A[i]) {
			swap[i] = min(swap[i], keep[i - 1] + 1); 
			keep[i] = min(keep[i], swap[i - 1]); // 可以通过交换前一个位置来同样实现递增.
		}
	}
	return min(swap.back(), keep.back());
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

/* 1155. Number of Dice Rolls With Target Sum */
/* You have d dice, and each die has f faces numbered 1, 2, ..., f. Return the number of 
* possible ways (out of fd total ways) modulo 10^9 + 7 to roll the dice so the sum of the 
* face up numbers equals target. Constraints: 1 <= d, f <= 30. 1 <= target <= 1000.
* Example: Input: d = 2, f = 6, target = 7. Output: 6
* Explanation:  You throw two dice, each with 6 faces.  There are 6 ways to get a sum of 7:
* 1+6, 2+5, 3+4, 4+3, 5+2, 6+1. */
// Bottom up dp method: 
int numRollsToTarget(int d, int target, int f, vector<vector<int>>& dp) {
	long res = 0; 
	if (d == 0 || target <= 0) return d == target; 
	if (dp[d][target] != -1) return dp[d][target];

	for (int i = 1; i <= f; ++i) {
		res = (res + numRollsToTarget(d - 1, target - i, i, dp)) % (1000000007);
	}
	return dp[d][target] = res;
}

int numRollsToTarget(int d, int f, int target) {
	vector<vector<int>> dp(31, vector<int>(1001, -1)); 
	return numRollsToTarget(d, target, f, dp);
}

/*
// Top down dp method:
int numRollsToTarget2(int d, int f, int target) {
	vector<vector<int>> dp(31, vector<int>(1001, -1));
	return dp.back().back(); 
}
*/

/* 920. Number of Music Playlists */
/* Your music player contains N different songs and she wants to listen to L
* (not necessarily different) songs during your trip.  You create a playlist
* so that: Every song is played at least once: A song can only be played again
* only if K other songs have been played. Return the number of possible playlists.
* As the answer can be very large, return it modulo 10^9 + 7.
* dp[i][j] denotes the solution of i songs with j different songs. return dp[L][N]. */
int numMusicPlaylists(int N, int L, int K) {
	long mod = 1e9 + 7;
	vector<vector<long>> dp(L + 1, vector<long>(N + 1, 0));
	dp[0][0] = 1;

	for (int i = 1; i <= L; ++i) {
		for (int j = 1; j <= N; ++j) {
			// Case 1: no changes since the last added one is new song.
			// the new song with the choices of N - (j - 1).
			dp[i][j] = (dp[i - 1][j - 1] * (N - (j - 1))) % mod;
			// Case 2: chose old songs. It should be updated j - k because k songs 
			// can't be chosed from j - 1 to j - k. If j <= K, this case will be 0.
			if (j > K) {
				dp[i][j] = (dp[i][j] + dp[i - 1][j] * (j - K)) % mod;
			}
		}
	}
	return dp.back().back();
}

/* 474. Ones and Zeroes */
/* suppose you are a dominator of m 0s and n 1s respectively. There is an array
* with strings consisting of only 0s and 1s. Now your task is to find the
* maximum number of strings that you can form with given m 0s and n 1s.
* Each 0 and 1 can be used at most once.
* Input: Array = {"10", "0001", "111001", "1", "0"}, m = 5, n = 3. Output: 4.
* “10,”0001”,”1”,”0”. */
int findMaxForm(vector<string>& strs, int m, int n) {
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

	for (auto s : strs) {
		int zero = 0, one = 0; 
		for (auto c : s) c == '1' ? ++one : ++zero;

		for (int i = m; i >= zero; --i) {
			for (int j = n; j >= one; --j) {
				dp[i][j] = max(dp[i][j], 1 + dp[i - zero][j - one]);
			}
		}
	}
	return dp.back().back(); 
}

/* Paint Fence */
/* There is a fence with n posts, each post can be painted with
* one of the k colors. You have to paint all the posts such that
* NO MORE THAN two adjacent fence posts have the same color.
* Return the total number of ways you can paint the fence. */
int numWays(int n, int k) {
	if (n <= 0) return 0;
	// "same": the last two posts have the same colors
	// "diff": the last two posts have different colors
	int diff = 0, same = k;

	for (int i = 2; i <= n; ++i) {
		int t = diff;
		diff = (same + diff) * (k - 1);
		same = t;
	}
	return diff + same;
}

/* 265. Paint House II -- HARD  */
/* There are a row of n houses, each house can be painted with one of the k colors. 
* The cost of painting each house with a certain color is different. You have to paint all the houses 
* such that no two adjacent houses have the same color.
* Paint houses using K colors, and no two adjacent has same color. Find the minimum cost to paint all.
* 用min1和min2来记录之前房子的最小和第二小的花费的颜色，如果当前房子颜色和min1相同，那么我们用min2对应的值计算，
* 反之我们用min1对应的值, 这种解法实际上也包含了求次小值的方法. */
int minCostII(vector<vector<int>>& costs) {
	if (costs.empty() || costs[0].empty()) return 0;
	vector<vector<int>> dp = costs;
	int min1 = -1, min2 = -1;

	for (int i = 0; i < dp.size(); ++i) {
		int last1 = min1, last2 = min2;
		min1 = -1; min2 = -1;
		for (int j = 0; j < dp[i].size(); ++j) {
			if (j != last1) {
				dp[i][j] += last1 < 0 ? 0 : dp[i - 1][last1];
			}
			else {
				dp[i][j] += last2 < 0 ? 0 : dp[i - 1][last2];
			}
			if (min1 < 0 || dp[i][j] < dp[i][min1]) {
				min2 = min1; min1 = j;
			}
			else if (min2 < 0 || dp[i][j] < dp[i][min2]) {
				min2 = j;
			}
		}
	}
	return dp.back()[min1];
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

/* 647. Palindromic Substrings */
/* Given a string, your task is to count how many palindromic substrings in this string.
 * The substrings with different start indexes or end indexes are counted as different substrings 
 * even they consist of same characters. Input: "abc". Output: 3. 
 * Explanation: Three palindromic strings: "a", "b", "c". */
int countSubstrings(string s) {
	int n = s.size(), res = 0; 
	vector<vector<int>> dp(n, vector<int>(n, 0));
	for (int i = n - 1; i >= 0; --i) {
		dp[i][i] = 1; 
		for (int j = i; j < n; ++j) {
			dp[i][j] = s[i] == s[j] && (j - i < 2 || dp[i + 1][j - 1]);
			if (dp[i][j]) ++res;
		}
	}
	return res; 
}

/* 416. Partition Equal Subset Sum */
/* Given a non-empty array containing only positive integers, find if the
* array can be partitioned into two subsets such that the sum of elements
* in both subsets is equal. */
bool canPartition(vector<int>& nums) {
	int sum = accumulate(nums.begin(), nums.end(), 0);
	if (sum % 2 != 0) return false;

	int target = sum / 2;
	vector<int> dp(target + 1, 0);
	dp[0] = 1;
	for (int i = 0; i < nums.size(); ++i) {
		for (int j = target; j >= nums[i]; --j) {
			dp[j] = dp[j] | dp[j - nums[i]];
		}
	}
	return dp.back();
}

/* 279. Perfect Squares */
/* Given a positive integer n, find the least number of perfect square
* numbers (for example, 1, 4, 9, 16, ...) which sum to n.
* Example 1: Input: n = 12.  Output: 3. Explanation: 12 = 4 + 4 + 4. */
int numSquares(int n) {
	vector<int> dp(n + 1, INT_MAX);
	dp[0] = 0; 
	for (int i = 0; i <= n; ++i) {
		for (int j = 0; j * j + i <= n; ++j) {
			dp[j * j + i] = min(1 + dp[i], dp[j * j + i]);
		}
	}
	return dp.back();
}

/* 486. Predict the Winner */
/* Given an array of scores that are non-negative integers.
* Player 1 picks one of the numbers from either end of the array
* followed by the player 2 and then player 1 and so on. Each time
* a player picks a number, that number will not be available for
* the next player. This continues until all the scores have been
* chosen. The player with the maximum score wins. Given an array
* of scores, predict whether player 1 is the winner. You can
* assume each player plays to maximize his score.
* dp[i][j] saves how much MORE scores that the first-in-action player
* will get from i to j THAN the second player.
* TOP DOWN DP, DP is used to save inter step results. */
int PredictTheWinner(vector<int>& nums, int i, int j, vector<vector<int> >& dp) {
	if (dp[i][j] == -1) {
		/* (1) Second player can get "PredictTheWinner(nums, i + 1, j, dp)" if first get "nums[i]"
		 * (2) Second player can get "PredictTheWinner(nums, i, j - 1, dp)" if first get "nums[j]" */
		dp[i][j] = (i == j) ? nums[i] : max(nums[i] - PredictTheWinner(nums, i + 1, j, dp),
			nums[j] - PredictTheWinner(nums, i, j - 1, dp));
	}
	return dp[i][j];
}

bool PredictTheWinner(vector<int>& nums) {
	int n = nums.size();
	vector<vector<int> > dp(n, vector<int>(n, -1));
	return PredictTheWinner(nums, 0, n - 1, dp) >= 0;
}

/* 304. Range Sum Query 2D - Immutable */
/* Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by 
* its upper left corner (row1, col1) and lower right corner (row2, col2). */
class NumMatrix {
public:
	NumMatrix(vector<vector<int>>& matrix) {
		if (matrix.empty() || matrix[0].empty()) return; 
		int m = matrix.size(), n = matrix[0].size();
		
		for (int i = 1; i <= m; ++i) {
			for (int j = 1; j <= n; ++j) {
				dp[i][j] += dp[i - 1][j] + dp[i][j - 1] - dp[i - 1][j - 1] + matrix[i - 1][j - 1]; 
			}
		}
	}
	int sumRegion(int row1, int col1, int row2, int col2) {
		return dp[row2 + 1][col2 + 1] - dp[row2 + 1][col1] - dp[row1][col2 + 1] + dp[row1][col1];
	}

private:
	vector<vector<int>> dp;
};

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
				dp[i][j] = (i > 0 && dp[i - 1][j] && (s[i - 1] == p[j - 2] || p[j - 2] == '.')) || dp[i][j-2];
			}
			else {
				dp[i][j] = i > 0 && dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
			}
		}
	}
	return dp.back().back(); 
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

/* 805. Split Array With Same Average -- HARD */
/* In a given integer array A, we must move every element of A to either list B or list C. 
* (B and C initially start empty.) Return true if and only if after such a move, 
* it is possible that the average value of B is equal to the average value of C, and B and C 
* are both non-empty. Example: Input: [1,2,3,4,5,6,7,8]. Output: true. */
bool splitArraySameAverage(vector<int>& A) {
	int sum = accumulate(A.begin(), A.end(), 0), n = A.size(), k = n / 2; 
	// 快速的剪枝过程. 
	bool possible = false; 
	for (int i = 1; i <= k; ++i) {
		if (!possible && sum * i % n == 0) possible = true; 
	}
	if (!possible) return false; 
	// dp[i] 表示数组中任选 i 个数字，所有可能的数字和。
	vector<unordered_set<int>> dp(k + 1);
	dp[0].insert(0); 
	// 更新 dp[i] 的思路是，对于 dp[i-1] 中的每个数字，都加上一个新的数字.
	for (auto a : A) {
		for (int i = k; i >= 1; --i) {
			for (auto b : dp[i - 1]) {
				dp[i].insert(a + b);
			}
		}
	}
	// 整个 dp 数组更新好了之后，下面就是验证的环节了.
	for (int i = 1; i <= k; ++i) {
		if (sum * i % n == 0 && dp[i].count(sum * i / n)) return true; 
	}
	return false; 
}

/* 877. Stone Game */
/* Alex and Lee play a game with piles of stones.  There are an even number of piles arranged in a row, 
* and each pile has a positive integer number of stones piles[i]. The objective of the game is to end 
* with the most stones.  The total number of stones is odd, so there are no ties. Assuming 
* Alex and Lee play optimally, return True if and only if Alex wins the game. Input: [5,3,4,5]. Output: true. */
int stoneGame(vector<int>& piles, vector<vector<int>>& dp, int i, int j) {
	if (dp[i][j] == -1) {
		dp[i][j] = i == j ? max(piles[i], piles[j]) : max(stoneGame(piles, dp, i + 1, j), stoneGame(piles, dp, i, j - 1));
	}
	return dp[i][j];
}

bool stoneGame(vector<int>& piles) {
	int n = piles.size();
	vector<vector<int>> dp(n, vector<int>(n, -1));
	return stoneGame(piles, dp, 0, n - 1) > 0;
}

/* 887. Super Egg Drop -- DP. */
/* You are given K eggs, and you have access to a building with N floors
* from 1 to N. Each egg is identical in function, and if an egg breaks,
* you cannot drop it again. You know that there exists a floor F with
* 0 <= F <= N such that any egg dropped at a floor higher than F will break,
* and any egg dropped at or below floor F will not break.
* Each move, you may take an egg (if you have an unbroken one) and
* drop it from any floor X (with 1 <= X <= N).
* Your goal is to know with certainty what the value of F is.
* What is the minimum number of moves that you need to know with
* certainty what F is, regardless of the initial value of F? */
/* LOGIC: dp[N][K]means that, given K eggs and N moves,
* what is the maximum number of floor that we can check.
* The dp equation is: dp[n][k] = dp[n - 1][k - 1] + dp[n - 1][k] + 1,
* which means we take 1 move to a floor,
* if egg breaks, then we can check dp[n - 1][k - 1] floors.
* if egg doesn't breaks, then we can check dp[n - 1][k] floors.
* Time Complexity: For time, O(NK) decalre the space, O(KlogN) running,
* For space, O(NK). */
int superEggDrop(int K, int N) {
	vector<vector<int>> dp(N + 1, vector<int>(K + 1, 0));
	int n = 0;
	while (dp[n][K] < N) {
		++n;
		for (int k = 1; k <= K; ++k) {
			dp[n][k] = 1 + dp[n - 1][k - 1] + dp[n - 1][k];
		}
	}
	return n;
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



// ===========================================================

// =================== 2. GREEDY PROBLMES ====================
/* 122. Best Time to Buy and Sell Stock II */
/* Say you have an array for which the ith element is the price of a given stock on day i.
* Design an algorithm to find the maximum profit. You may complete as many transactions 
* as you like (i.e., buy one and sell one share of the stock multiple times). 
* Note: You may not engage in multiple transactions at the same time (i.e., you must sell 
* the stock before you buy again). */
int maxProfit2(vector<int>& prices) {
	if (prices.empty()) return 0;
	int res = 0;
	for (int i = 0; i < prices.size() - 1; ++i) {
		if (prices[i + 1] > prices[i]) res += prices[i + 1] - prices[i];
	}
	return res;
}

/* 948. Bag of Tokens -- TWO POINTERS */
/* You have an initial power P, an initial score of 0 points, and a bag of tokens.
* Each token can be used at most once, has a value token[i], and has potentially
* two ways to use it.(1) If we have at least token[i] power, we may play the token face up, 
* losing token[i] power, and gaining 1 point. (2) If we have at least 1 point, we may 
* play the token face down, gaining token[i] power, and losing 1 point. Return the 
* largest number of points we can have after playing any number of tokens. 
* Input: tokens = [100,200,300,400], P = 200. Output: 2.
* Input: tokens = [100,200], P = 150. Output: 1. */
int bagOfTokensScore(vector<int>& tokens, int P) {
	int res = 0, n = tokens.size(), i = 0, j = n - 1, score = 0; 
	sort(tokens.begin(), tokens.end());
	while (i <= j) {
		if (P >= tokens[i]) {
			P -= tokens[i++];
			res = max(res, ++score);
		}
		else if (score) {
			--score;
			P += tokens[j--];
		}
		else {
			break; 
		}
	}
	return res; 
}

/* 419. Battleships in a Board -- NOT REALLY GREEDY */
/* Given an 2D board, count how many battleships are in it. The battleships are
* represented with 'X's, empty slots are represented with '.'s. You may assume 
* the following rules: You receive a valid board, made of only battleships or empty slots.
* Battleships can only be placed horizontally or vertically. In other words, 
* they can only be made of the shape 1xN (1 row, N columns) or Nx1 (N rows, 1 column),
* where N can be of any size. At least one horizontal or vertical cell separates 
* between two battleships - there are no adjacent battleships. */
int countBattleships(vector<vector<char>>& board) {
	int m = board.size(), n = board[0].size(), res = 0;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (board[i][j] == '.' || (i > 0 && board[i][j] == 'X' && board[i - 1][j] == 'X') || (j > 0 && board[i][j] == 'X' && board[i][j - 1] == 'X')) continue;
			++res;
		}
	}
	return res;
}

/* 881. Boats to Save People */
/* The i-th person has weight people[i], and each boat can carry a maximum weight 
* of limit. Each boat carries at most 2 people at the same time, provided the sum 
* of the weight of those people is at most limit. Return the minimum number of boats 
* to carry every given person. Input: people = [3,2,2,1], limit = 3. Output: 3*/
int numRescueBoats(vector<int>& people, int limit) {
	int i, j;
	sort(people.rbegin(), people.rend());

	for (i = 0, j = people.size() - 1; i <= j; ++i)
		if (people[i] + people[j] <= limit) j--;

	return i;
}

/* 1057. Campus Bikes -- BUCKET SORT */
/* On a campus represented as a 2D grid, there are N workers and M bikes, with N <= M. 
* Each worker and bike is a 2D coordinate on this grid. Return a vector ans of length N, 
* where ans[i] is the index (0-indexed) of the bike that the i-th worker is assigned to.*/
vector<int> assignBikes(vector<vector<int>>& workers, vector<vector<int>>& bikes) {
	int n = workers.size(), m = bikes.size(); 
	vector<int> res(n, -1);
	// buckets[i] is the vector<worker id, bike id> with distance i
	vector<vector<pair<int, int>>> bucket(2001); 

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			int dist = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1]);
			bucket[dist].push_back({ i, j });
		}
	}
	unordered_set<int> bikeUsed; 
	for (int i = 0; i <= 2000; ++i) {
		for (int j = 0; j < bucket[i].size(); ++j) {
			if (res[bucket[i][j].first] == -1 && !bikeUsed.count(bucket[i][j].second)) {
				bikeUsed.insert(bucket[i][j].second); 
				res[bucket[i][j].first] = bucket[i][j].second; 
			}
		}
	}
	return res; 
}

/* 1066. Campus Bikes II -- BRUTE FORCE & BACKTRACKING */
/* On a campus represented as a 2D grid, there are N workers and M bikes,
* with N <= M. Each worker and bike is a 2D coordinate on this grid.
* We assign one unique bike to each worker so that the sum of the
* Manhattan distances between each worker and their assigned bike is MINIMIZED.*/
// Solution 1: brute force with some pruning. 
int calManDist(vector<int>& A, vector<int>& B) {
	return abs(A[0] - B[0]) + abs(A[1] - B[1]);
}

void assignBikes2(vector<vector<int>>& workers, vector<vector<int>>& bikes,
	vector<int>& visited, int cur, int dist, int& res) {
	if (cur >= workers.size()) {
		res = min(res, dist);
		return;
	}
	if (dist > res) return;
	for (int i = 0; i < bikes.size(); ++i) {
		if (visited[i] == 0) {
			visited[i] = 1;
			assignBikes2(workers, bikes, visited, cur + 1, dist + calManDist(workers[cur], bikes[i]), res);
			visited[i] = 0;
		}
	}
}

int assignBikes2(vector<vector<int>>& workers, vector<vector<int>>& bikes) {
	int res = INT_MAX;
	vector<int> visited(bikes.size(), 0);
	assignBikes2(workers, bikes, visited, 0, 0, res);
	return res;
}

/* 135. Candy -- HARD */
/* There are N children standing in a line. Each child is assigned a rating value.
* You are giving candies to these children subjected to the following requirements:
* Each child must have at least one candy.
* Children with a higher rating get more candies than their neighbors.
* What is the minimum candies you must give? Input: [1,0,2]. Output: 4. 
* Logic: two traverse, from to end & end to front. */
int candy(vector<int>& ratings) {
	int n = ratings.size(), res = 0; 
	vector<int> v(n, 1);
	for (int i = 0; i < n - 1; ++i) {
		if (ratings[i + 1] > ratings[i]) v[i + 1] = v[i] + 1; 
	}
	for (int i = n - 2; i >= 0; --i) {
		if (ratings[i] > ratings[i + 1]) v[i] = max(v[i], v[i + 1] + 1);
	}
	for (auto a : v) res += a; 
	return res; 
}

/* 853. Car Fleet */
/* N cars are going to the same destination along a one lane road.  The destination is target 
* miles away. Each car i has a constant speed speed[i] (in miles per hour), and initial position 
* position[i] miles towards the target along the road. 
* 只关心是否能组成车队一同经过终点线，那么如何才能知道是否能一起过线呢，最简单的方法就是看时间，
* 假如车B在车A的后面，而车B到终点线的时间小于等于车A，那么就知道车A和B一定会组成车队一起过线。*/
int carFleet(int target, vector<int>& position, vector<int>& speed) {
	map<int, double> m;
	int n = position.size();
	for (int i = 0; i < n; ++i) {
		m[-position[i]] = (double)(target - position[i]) / speed[i];
	}

	double mx = 0;
	int res = 0;
	for (auto it : m) {
		if (it.second > mx) {
			++res;
			mx = it.second;
		}
	}
	return res;
}

/* 765. Couples Holding Hands */
/* The people and seats are represented by an integer from 0 to 2N-1, the couples are numbered in order,
* the first couple being (0, 1), the second couple being (2, 3), and so on with the last couple being 
* (2N-2, 2N-1). The couples' initial seating is given by row[i] being the value of the person who is 
* initially sitting in the i-th seat. */
int minSwapsCouples(vector<int>& row) {
	int res = 0, n = row.size();
	for (int i = 0; i < n - 1; i += 2) {
		if (row[i + 1] == (row[i] ^ 1)) continue;
		++res;
		for (int j = i + 1; j < n; ++j) {
			if (row[j] == (row[i] ^ 1)) {
				row[j] = row[i + 1];
				row[i + 1] = (row[i] ^ 1);
				break;
			}
		}
	}
	return res;
}

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

/* Find Permutation */
/* Input: "DI". Output: [2,1,3]. Explanation: Both [2,1,3] and [3,1,2] can construct the secret signature "DI", 
but since we want to find the one with the smallest lexicographical permutation, you need to output [2,1,3] */
vector<int> findPermutation(string s) {
	int n = s.size(); 
	vector<int> res(n + 1, 0);
	for (int i = 0; i <= n; ++i) res[i] = i + 1; 
	for (int i = 0; i < n; ++i) {
		if (s[i] == 'D') {
			int j = i;
			while (j < n && s[j] == s[i]) ++j;
			reverse(res.begin() + i, res.begin() + j + 1); 
			i = j; // IMPORTANT.
		}
	}
	return res;
}

/* 134. Gas Station */
/* There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
* You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i 
* to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.
* Return the starting gas station's index if you can travel around the circuit once in the 
* clockwise direction, otherwise return -1. */
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
	int n = gas.size(), sum = 0, total = 0, start = 0; 
	for (int i = 0; i < n; ++i) {
		sum += gas[i] - cost[i]; 
		total += gas[i] - cost[i]; 

		if (sum < 0) {
			start = i + 1;
			sum = 0;
		}
	}
	if (total < 0) return -1; 
	return start; 
}

/* 55. Jump Game */
/* Given an array of non-negative integers, you are initially positioned at the first index of the array.
* Each element in the array represents your maximum jump length at that position. Determine if you are 
* able to reach the last index. Input: [2,3,1,1,4]. Output: true. */
bool canJump(vector<int>& nums) {
	int n = nums.size(); 
	vector<int> dp(n + 1, 0);
	dp[0] = nums[0]; 
	// IMPORTANT: the last element is not included
	for (int i = 1; i < n; ++i) {
		dp[i] = max(dp[i - 1], nums[i - 1]) - 1; 
		if (dp[i] < 0) return false; 
	}
	return dp.back() >= 0; 
}

/* 45. Jump Game II -- IMPORTANT */
/* Given an array of non-negative integers, you are initially positioned at the first index of the array.
* Each element in the array represents your maximum jump length at that position.
* Your goal is to reach the last index in the minimum number of jumps. Input: [2,3,1,1,4]. Output: 2. */
int jump(vector<int>& nums) {
	int n = nums.size(), res = 0, cur = 0, i = 0;
	while (cur < n - 1) {
		++res;
		int pre = cur;
		for (; i <= pre; ++i) {
			cur = max(cur, i + nums[i]);
		}
	}
	return res;
}

/* 689. Maximum Sum of 3 Non-Overlapping Subarrays */
/* In a given array nums of positive integers, find three non-overlapping
* subarrays with maximum sum. Each subarray will be of size k, and we want
* to maximize the sum of all 3*k entries. Return the result as a list of
* indices representing the starting position of each interval (0-indexed).
* If there are multiple answers, return the lexicographically smallest one.
* LOGIC: If the middle interval is [i, i+k-1], where k <= i <= n-2k,
* the left interval has to be in subrange [0, i-1],
* and the right interval is from subrange [i+k, n-1]. */
vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k) {
	int n = nums.size(), mx = INT_MIN;
	vector<int> sums{ 0 }, res, left(n, 0), right(n, n - k);
	for (int num : nums) sums.push_back(sums.back() + num);

	for (int i = k, total = sums[k] - sums[0]; i < n; ++i) {
		if (sums[i + 1] - sums[i + 1 - k] > total) {
			left[i] = i + 1 - k;
			total = sums[i + 1] - sums[i + 1 - k];
		}
		else {
			left[i] = left[i - 1];
		}
	}
	for (int i = n - 1 - k, total = sums[n] - sums[n - k]; i >= 0; --i) {
		if (sums[i + k] - sums[i] >= total) {
			right[i] = i;
			total = sums[i + k] - sums[i];
		}
		else {
			right[i] = right[i + 1];
		}
	}
	// test every possible starting index of middle interval, i.e. k <= i <= n-2k
	for (int i = k; i <= n - 2 * k; ++i) {
		int l = left[i - 1], r = right[i + k];
		int total = (sums[i + k] - sums[i]) + (sums[l + k] - sums[l]) + (sums[r + k] - sums[r]);
		if (mx < total) {
			mx = total;
			res = { l, i, r };
		}
	}
	return res;
}

/* 1007. Minimum Domino Rotations For Equal Row */
/* In a row of dominoes, A[i] and B[i] represent the top and bottom halves of the i-th domino.  
* (A domino is a tile with two numbers from 1 to 6 - one on each half of the tile.)
* We may rotate the i-th domino, so that A[i] and B[i] swap values.
* Return the minimum number of rotations so that all the values in A are the same, 
* or all the values in B are the same. If it cannot be done, return -1. 
* Logic: Count the occurrence of all numbers in A and B, and also the number of domino with two same numbers.
* Try all possibilities from 1 to 6. If we can make number i in a whole row, it should satisfy that countA[i] + countB[i] - same[i] = n */
int minDominoRotations(vector<int>& A, vector<int>& B) {
	vector<int> countA(7), countB(7), same(7); 
	int res = 6, n = A.size();
	for (int i = 0; i < n; ++i) {
		++countA[A[i]];
		++countB[B[i]];
		if (A[i] == B[i]) ++same[A[i]];
	}
	for (int i = 1; i <= 6; ++i) {
		if (countA[i] + countB[i] - same[i] == n) {
			res = min(res, n - max(countA[i], countB[i]));
		}
	}
	return res; 
}

/* 330. Patching Array */
/* Given a sorted positive integer array nums and an integer n, add/patch elements to 
* the array such that any number in range [1, n] inclusive can be formed by the sum 
* of some elements in the array. Return the minimum number of patches required. 
* Input: nums = [1,3], n = 6. Output: 1. 
* LOGIC: "miss" 用来表示[0,n]之间最小的不能表示的值. Initiate with 1.  */
int minPatches(vector<int>& nums, int n) {
	int res = 0, i = 0, len = nums.size();
	long long miss = 1;
	while (miss <= n) {
		if (i < len && nums[i] <= miss) {
			miss += nums[i++];
		}
		else {
			miss += miss;
			++res;
		}
	}
	return res;
}

/* 402. Remove K Digits */
/* Given a non-negative integer num represented as a string, remove k digits from the number 
* so that the new number is the smallest possible. Note: The length of num is less than 10002 
* and will be ≥ k. The given num does not contain any leading zero.
* Example 1: Input: num = "1432219", k = 3. Output: "1219". */
string removeKdigits(string num, int k) {
	int n = num.size(), keep = n - k; 
	string res("");
	for (auto c : num) {
		while (res.back() > c && k > 0) {
			res.pop_back(); 
			--k;
		}
		res += c;
	}
	res.resize(keep);
	while (res.size() && res[0] == '0') res.erase(res.begin());
	return res.empty() ? "0" : res;
}

/* 1055. Shortest Way to Form String */
/* From any string, we can form a subsequence of that string by deleting some number of characters 
* (possibly no deletions). Given two strings source and target, return the minimum number of 
* subsequences of source such that their concatenation equals target. If the task is impossible, return -1.
* Input: source = "abc", target = "abcbc". Output: 2. Input: source = "abc", target = "acdbc". Output: -1. */
int shortestWay(string source, string target) {
	int res = 0, n = target.size();
	for (int i = 0; i < n; ) {
		int start = i;
		for (int j = 0; j < source.size(); ++j) {
			if (source[j] == target[i]) ++i;
		}
		if (i == start) return -1;
		++res;
	}
	return res;
}

// ===========================================================

// ==================== 3. DFS PROBLEMS ======================

/* 104. Maximum Depth of Binary Tree */
int maxDepth(TreeNode* root) {
	if (!root) return 0;
	return 1 + max(maxDepth(root->left), maxDepth(root->right));
}

/* 559. Maximum Depth of N-ary Tree */
/* Given a n-ary tree, find its maximum depth. The maximum
* depth is the number of nodes along the longest path from
* the root node down to the farthest leaf node. */
// IMPORTANT. "int& res" PASS BY REFERENCE. 
void maxDepthNaryTree(NaryTreeNode* node, int ind, int& res) {
	if (!node) return;
	res = max(res, ind);

	for (auto a : node->children) {
		maxDepthNaryTree(a, ind + 1, res);
	}
}

int maxDepthNaryTree(NaryTreeNode* root) {
	if (!root) return 0;
	int res = 0;
	maxDepthNaryTree(root, 1, res);
	return res;
}

/* 257. Binary Tree Paths */
/* Given a binary tree, return all root-to-leaf paths. */
void binaryTreePaths(TreeNode* node, string ind, vector<string>& res) {
	ind += to_string(node->val);
	if (!node->left && !node->right) {
		res.push_back(ind);
		return;
	}
	if (node->left) binaryTreePaths(node->left, ind + "->", res);
	if (node->right) binaryTreePaths(node->right, ind + "->", res);
}

vector<string> binaryTreePaths(TreeNode* root) {
	vector<string> res;
	if (!root) return res;
	binaryTreePaths(root, "", res);
	return res;
}

/* 988. Smallest String Starting From Leaf -- Reverse tree traversal */
/* Given the root of a binary tree, each node has a value from 0 to 25 representing the letters 'a' to 'z': 
* a value of 0 represents 'a', a value of 1 represents 'b', and so on. Find the lexicographically
* smallest string that starts at a leaf of this tree and ends at the root. */
string smallestFromLeaf(TreeNode* node, string s) {
	if (!node) return "|"; // "|" is greater in value than 'z'.
	s = string(1, node->val + 'a') + s;
	return node->left == node->right ? s : min(smallestFromLeaf(node->left, s), smallestFromLeaf(node->right, s));
}

string smallestFromLeaf(TreeNode* root) {
	string res("");
	return smallestFromLeaf(root, res);
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

/* 226. invertTree */
TreeNode* invertTree(TreeNode* root) {
	if (!root) return NULL;
	TreeNode* t = root->left;
	root->left = invertTree(root->right);
	root->right = invertTree(t);
	return root;
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

/* 450. Delete Node in a BST */
/* Given a root node reference of a BST and a key, delete the node with the given key in the BST. 
* Return the root node reference (possibly updated) of the BST. */
TreeNode* deleteNode(TreeNode* root, int key) {
	if (!root) return NULL; 
	TreeNode* p = root; 
	if (p->val > key) p->left = deleteNode(root->left, key);
	else if (p->val < key) p->right = deleteNode(root->right, key);
	else {
		if (!p->left || !p->right) {
			return p->left ? p->left : p->right; 
		}
		else {
			TreeNode* t = p->right; 
			while (t->left) t = t->left; 
			p->val = t->val; 
			p->right = deleteNode(p->right, t->val);
		}
	}
	return root; 
}

/* 783. Minimum Distance Between BST Nodes */
/* Given a Binary Search Tree (BST) with the root node root, return the minimum difference 
* between the values of any two different nodes in the tree. */
int minDiffInBST(TreeNode* root) {
	int res = INT_MAX, pre = 0;
	if (!root) return res; 
	stack<TreeNode*> st; 
	TreeNode* p = root; 

	while (!st.empty() || p) {
		while (p) {
			st.push(p); 
			p = p->left; 
		}
		p = st.top(); st.pop(); 
		if (pre != -1) res = min(res, abs(p->val - pre));
		pre = p->val; 
		p = p->right; 
	}
	return res; 
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

/* 144. Binary Tree Preorder Traversal */
vector<int> preorderTraversal(TreeNode* root) {
	vector<int> res;
	if (!root) return res;
	stack<TreeNode*> st{ { root } };

	while (!st.empty()) {
		auto t = st.top(); st.pop();
		res.push_back(t->val);

		if (t->right) st.push(t->right);
		if (t->left) st.push(t->left);
	}

	return res;
}

/* 145. Binary Tree Postorder Traversal -- HARD */
/* Given a binary tree, return the postorder traversal of its nodes' values. */
vector<int> postorderTraversal(TreeNode* root) {
	vector<int> res;
	if (!root) return res;
	stack<TreeNode*> st;
	st.push(root);
	TreeNode* child = root;

	while (!st.empty()) {
		auto t = st.top();
		if ((!t->left && !t->right) || t->left == child || t->right == child) {
			res.push_back(t->val);
			child = t;
			st.pop();
		}
		else {
			if (t->right) st.push(t->right);
			if (t->left) st.push(t->left);
		}
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
		m[t.second].push_back(t.first -> val);
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

/* 938. Range Sum of BST */
/* Given the root node of a binary search tree, return the sum of values of all nodes 
* with value between L and R (inclusive). The binary search tree is guaranteed to have unique values. */
void rangeSumBSTInorder(TreeNode* node, int L, int R, int& sum) {
	if (!node) return;
	rangeSumBSTInorder(node->left, L, R, sum);
	if (node->val >= L && node->val <= R) sum += node->val;
	rangeSumBSTInorder(node->right, L, R, sum);
}

int rangeSumBST(TreeNode* root, int L, int R) {
	if (!root) return 0;
	int sum = 0;
	rangeSumBSTInorder(root, L, R, sum);
	return sum;
}

/* 530. Minimum Absolute Difference in BST */
/* Given a binary search tree with non-negative values, 
* find the minimum absolute difference between values of any two nodes. */
void inorderHelper(TreeNode* node, vector<int>& v) {
	if (node->left) inorderHelper(node->left, v);
	v.push_back(node->val);
	if (node->right) inorderHelper(node->right, v);
}

int getMinimumDifference(TreeNode* root) {
	if (!root) return 0;
	vector<int> v;
	inorderHelper(root, v);
	int res = INT_MAX;
	for (int i = 0; i < v.size() - 1; ++i) {
		res = min(res, abs(v[i + 1] - v[i]));
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

/* 872. Leaf-Similar Trees */
/* Consider all the leaves of a binary tree.
* From left to right order, the values of those
* leaves form a leaf value sequence.
* Two binary trees are considered leaf-similar
* if their leaf value sequence is the same.
* Return true if and only if the two given trees
* with head nodes root1 and root2 are leaf-similar.*/
void getLeaves(TreeNode* root, vector<int>& res) {
	if (!root) return;
	if (!root->left && !root->right) {
		res.push_back(root->val);
	}
	if (root->left) getLeaves(root->left, res);
	if (root->right) getLeaves(root->right, res);
}

bool leafSimilar(TreeNode* root1, TreeNode* root2) {
	if (!root1 || !root2) return false;
	vector<int> v1, v2;
	getLeaves(root1, v1);
	getLeaves(root2, v2);
	return v1 == v2;
}

/* 701. Insert into a Binary Search Tree */
/* Given the root node of a binary search tree (BST) and a value to be inserted into the tree, 
* insert the value into the BST. Return the root node of the BST after the insertion. 
* It is guaranteed that the new value does not exist in the original BST. */
TreeNode* insertIntoBST(TreeNode* root, int val) {
	if (!root) return NULL; 
	if (root->val > val) root->left = insertIntoBST(root->left, val);
	else if (root ->val < val) root->right = insertIntoBST(root->right, val);
	return root; 
}

/* 951. Flip Equivalent Binary Trees */
/* A binary tree X is flip equivalent to a binary tree Y if and only if we can 
* make X equal to Y after some number of flip operations. */
bool flipEquiv(TreeNode* root1, TreeNode* root2) {
	if ((!root1 && root2) || (root1 && !root2)) return false;
	if (root1) {
		return ((root1->val == root2->val) &&
			((flipEquiv(root1->left, root2->right) && flipEquiv(root1->right, root2->left)) ||
			(flipEquiv(root1->left, root2->left) && flipEquiv(root1->right, root2->right))));
	}
	return true;
}

/* 501. Find Mode in Binary Search Tree */
/* find all the mode(s) (the most frequently occurred element) in the given BST. */
vector<int> findMode(TreeNode* root) {
	vector<int> res;
	if (!root) return res;
	int mx = INT_MIN;
	unordered_map<int, int> m;
	inorder(root, m, mx);

	for (auto it : m) {
		if (it.second == mx) res.push_back(it.first);
	}

	return res;
}

void inorder(TreeNode* node, unordered_map<int, int>& m, int& mx) {
	if (!node) return;
	inorder(node->left, m, mx);
	mx = max(mx, ++m[node->val]);
	inorder(node->right, m, mx);
}

/* 687. Longest Univalue Path */
/* Given a binary tree, find the length of the longest path where each node in the path has the same value. 
* This path may or may not pass through the root. The length of path between two nodes is represented by 
* the number of edges between them. */
/* 需要注意的是我们的递归函数helper返回值的意义，并不是经过某个结点的最长路径的长度，最长路径长度保存在了结果res中，
* 不是返回值，返回的是以该结点为终点的最长路径长度，这样回溯的时候，我们还可以继续连上其父结点 */
int longestUnivaluePath(TreeNode* node, int& res) {
	if (!node) return 0;
	int left = longestUnivaluePath(node->left, res);
	int right = longestUnivaluePath(node->right, res);
	left = (node->left && node->left->val == node->val) ? 1 + left : 0; 
	right = (node->right && node->right->val == node->val) ? 1 + right : 0; 
	res = max(res, left + right); 
	return max(left, right);
}

int longestUnivaluePath(TreeNode* root) {
	if (!root) return 0; 
	int res = 0; 
	longestUnivaluePath(root, res);
	return res; 
}

/* 545. Boundary of Binary Tree */
/* Given a binary tree, return the values of its boundary in anti-clockwise direction 
* starting from root. Boundary includes left boundary, leaves, and right boundary 
* in order without duplicate nodes.  */
void leftBoundary(TreeNode* node, vector<int>& res) {
	if (!node || (!node->left && !node->right)) return;
	res.push_back(node->val);
	if (node->left) leftBoundary(node->left, res);
	else leftBoundary(node->right, res);
}

void leaves(TreeNode* node, vector<int>& res) {
	if (!node) return;
	if (!node->left && !node->right) res.push_back(node->val);
	if (node->left) leaves(node->left, res);
	if (node->right) leaves(node->right, res);
}

void rightBoundary(TreeNode* node, vector<int>& res) {
	if (!node || (!node->left && !node->right)) return;
	if (node->right) rightBoundary(node->right, res);
	else rightBoundary(node->left, res);
	res.push_back(node->val);
}

vector<int> boundaryOfBinaryTree(TreeNode* root) {
	vector<int> res;
	if (!root) return res;
	if (root->left || root->right) res.push_back(root->val);

	leftBoundary(root->left, res);
	leaves(root, res);
	rightBoundary(root->right, res);

	return res;
}

/* 366. Find Leaves of Binary Tree -- 剥洋葱方法  */
TreeNode* findLeaves(TreeNode* node, vector<int>& ind) {
	if (!node) return NULL;
	if (!node->left && !node->right) {
		ind.push_back(node->val);
		return NULL;
	}
	node->left = findLeaves(node->left, ind);
	node->right = findLeaves(node->right, ind);
	return node;
}

vector<vector<int>> findLeaves(TreeNode* root) {
	vector<vector<int>> res;
	if (!root) return res;
	while (root) {
		vector<int> ind;
		root = findLeaves(root, ind);
		res.push_back(ind);
	}
	return res;
}

/* 285. Inorder Successor in BST */
/* Given a binary search tree and a node in it, find the in-order successor of that node in the BST.
* The successor of a node p is the node with the smallest key greater than p.val. */
TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
	if (!root) return NULL;
	stack<TreeNode*> st;
	TreeNode* t = root;
	bool found = false;

	while (!st.empty() || t) {
		while (t) {
			st.push(t);
			t = t->left;
		}
		t = st.top(); st.pop();
		if (found) return t;
		if (t->val == p->val) found = true;
		t = t->right;
	}
	return NULL;
}

/* 261. Graph Valid Tree */
/* Given n nodes labeled from 0 to n-1 and a list of
* undirected edges (each edge is a pair of nodes),
* write a function to check whether these edges
* make up a valid tree.
* TEST (1) NO CYCLE. (2) ALL NODES VISITED. */
bool hasCycle(vector<vector<int>>& g, vector<int>& visited, int cur, int pre) {
	if (visited[cur]) return true;
	visited[cur] = 1;
	for (auto a : g[cur]) {
		if (a == pre) continue;
		if (hasCycle(g, visited, a, cur)) return true;
	}
	return false;
}

bool validTree(int n, vector<vector<int>>& edges) {
	vector<vector<int>> g(n, vector<int>());
	for (auto a : edges) {
		g[a[0]].push_back(a[1]);
		g[a[1]].push_back(a[0]);
	}
	vector<int> visited(n, 0);
	if (hasCycle(g, visited, 0, -1)) return false;
	for (int i = 0; i < n; ++i) {
		if (!visited[i]) return false;
	}
	return true;
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

/* 536. Construct Binary Tree from String */
/* The whole input represents a binary tree. It contains an integer followed by zero, 
* one or two pairs of parenthesis. The integer represents the root's value and a pair 
* of parenthesis contains a child binary tree with the same structure.
* You always start to construct the left child node of the parent first if it exists. 
* Input: "4(2(3)(1))(6(5))". */
TreeNode* str2tree(string s) {
	if (s.empty()) return NULL;
	auto found = s.find('(');
	int val = found == string::npos ? stoi(s) : stoi(s.substr(0, found));
	TreeNode* root = new TreeNode(val);

	int cnt = 0, start = found;
	for (int i = start; i < s.size(); ++i) {
		if (s[i] == '(') ++cnt;
		else if (s[i] == ')') --cnt;

		if (cnt == 0 && start == found) {
			root->left = str2tree(s.substr(start + 1, i - start - 1));
			start = i + 1;
		}
		else if (cnt == 0) {
			root->right = str2tree(s.substr(start + 1, i - start - 1));
		}
	}
	return root;
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

/* 298. Binary Tree Longest Consecutive Sequence */
/* Given a binary tree, find the length of the longest consecutive sequence path
* The path refers to any sequence of nodes from some starting node to any node 
* in the tree along the parent-child connections. The longest consecutive path 
* need to be from parent to child (cannot be the reverse). */
void longestConsecutive(TreeNode* node, TreeNode* pre, int ind, int& res) {
	if (!node) return; 
	if (node->val == pre->val + 1) ++ind; 
	else ind = 1; 
	res = max(res, ind);
	longestConsecutive(node->left, node, ind, res);
	longestConsecutive(node->right, node, ind, res);
}

int longestConsecutive(TreeNode* root) {
	int res = 0;
	longestConsecutive(root, root, 1, res);
	return res;
}

/* 549. Binary Tree Longest Consecutive Sequence II */
/*  the path can be in the child-Parent-child order, where not necessarily be parent-child order. */
pair<int, int> longestConsecutive2(TreeNode* parent, TreeNode* child, int& res) {
	if (!child) return { 0, 0 };
	auto left = longestConsecutive2(child, child->left, res);
	auto right = longestConsecutive2(child, child->right, res);
	res = max(res, 1 + left.first + right.second);
	res = max(res, 1 + left.second + right.first);

	int inc = 0, dec = 0; 
	if (parent->val == child->val + 1) {
		dec = 1 + max(left.second, right.second);
	}
	else if (child->val == parent->val + 1) {
		inc = 1 + max(left.first, right.first);
	}
	return { inc, dec };
}

int longestConsecutive2(TreeNode* root) {
	if (!root) return 0;
	int res = 0; 
	longestConsecutive2(root, root, res);
	return res; 
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

/* 272. Closest Binary Search Tree Value II -- HARD */
/* Given a non-empty binary search tree and a target value, find k values in the BST that 
* are closest to the target. Given target value is a floating point. You may assume k is 
* always valid, that is: k ≤ total nodes. */
void closestKValues(TreeNode* node, double target, int k, vector<int>& res) {
	if (!node) return;
	closestKValues(node->left, target, k, res);
	if (res.size() < k) {
		res.push_back(node->val);
	}
	else if (abs(node->val - target) < abs(res[0] - target)) {
		res.erase(res.begin());
		res.push_back(node->val);
	}
	else {
		return;
	}
	closestKValues(node->right, target, k, res);
}

vector<int> closestKValues(TreeNode* root, double target, int k) {
	vector<int> res;
	if (!root) return res;
	closestKValues(root, target, k, res);
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

/* 133. Clone Graph */
/* Given a reference of a node in a connected undirected graph,
* return a deep copy (clone) of the graph. Each node in the graph
* contains a val (int) and a list (List[Node]) of its neighbors. */
NodeNeighbor* cloneGraph(NodeNeighbor* node, unordered_map<NodeNeighbor*, NodeNeighbor*>& m) {
	if (!node) return NULL;
	if (m.find(node) == m.end()) {
		m[node] = new NodeNeighbor(node->val);
		for (auto a : node->neighbors) {
			m[node]->neighbors.push_back(cloneGraph(a, m));
		}
	}
	return m[node];
}

NodeNeighbor* cloneGraph(NodeNeighbor* node) {
	unordered_map<NodeNeighbor*, NodeNeighbor*> m;
	return cloneGraph(node, m);
}

/* 679. 24 Game （找某个明确目标）*/
/* You have 4 cards each containing a number from 1 to 9.
* You need to judge whether they could operated
* through *, /, +, -, (, ) to get the value of 24.
* Input: [4, 1, 8, 7], Output: True.
* Explanation: (8-4) * (7-1) = 24. */
bool judgePoint24(vector<double>& vec, double& esp, bool& res) {
	if (vec.empty()) return res;
	if (vec.size() == 1) { // WHY not use "&&" ???
		if (abs(vec[0] - 24) < esp) res = true;
	}
	int n = vec.size();

	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			double p = vec[i], q = vec[j];
			vector<double> t = { p + q, p - q, q - p, p * q };
			if (q) t.push_back(p / q);
			if (p) t.push_back(q / p);
			// IMPORTANT: the order matters
			vec.erase(vec.begin() + j);
			vec.erase(vec.begin() + i);

			for (auto a : t) {
				vec.push_back(a);
				// DFS
				judgePoint24(vec, esp, res);
				vec.pop_back();
			}
			// IMPORTANT: the order matters
			vec.insert(vec.begin() + i, p);
			vec.insert(vec.begin() + j, q);
		}
	}
	return res;
}

bool judgePoint24(vector<int>& nums) {
	double esp = 10e-8;
	bool res = false;
	vector<double> vec(nums.begin(), nums.end());
	return judgePoint24(vec, esp, res);
}

/* 797. All Paths From Source to Target -- DFS & BACKTRACK */
/* Given a directed, acyclic graph of N nodes.  Find all possible paths from 
* node 0 to node N-1, and return them in any order. graph[i] is a list of 
* all nodes j for which the edge (i, j) exists.
* Input: [[1,2], [3], [3], []]. Output: [[0,1,3],[0,2,3]]. */
void allPathsSourceTarget(vector<vector<int>>& g, int ix, vector<int>& ind, vector<vector<int>>& res) {
	ind.push_back(ix);
	int n = g.size();
	if (ix == n - 1) {
		res.push_back(ind);
		return;
	}
	for (auto a : g[ix]) {
		allPathsSourceTarget(g, ix + 1, ind, res);
	}
	ind.pop_back(); 
}

vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
	vector<vector<int>> res; 
	vector<int> ind; 
	allPathsSourceTarget(graph, 0, ind, res);
	return res; 
}

/* 1059. All Paths from Source Lead to Destination -- 染色法 */
/* Given the edges of a directed graph, and two nodes
* source and destination of this graph, determine
* whether or not ALL paths starting from source
* eventually end at destination.
* Input: n = 4, edges = [[0,1],[0,3],[1,2],[2,1]],
* source = 0, destination = 3. Output: false.  */
bool leadsToDestination(unordered_map<int, vector<int>>& m, vector<int>& visited, int cur, int destination) {
	if (m[cur].empty()) return cur == destination;
	if (visited[cur] != -1) return visited[cur];
	visited[cur] = 0;

	for (auto a : m[cur]) {
		if (!leadsToDestination(m, visited, a, destination)) return false;
	}
	return visited[cur] = true;
}

bool leadsToDestination(int n, vector<vector<int>>& edges, int source, int destination) {
	unordered_map<int, vector<int>> m;
	for (auto a : edges) {
		m[a[0]].push_back(a[1]);
	}
	// -1 means not visited
	// 0 means return false
	// 1 means true statement
	vector<int> visited(n, -1);
	return leadsToDestination(m, visited, source, destination);
}

/* 968. Binary Tree Cameras-- HARD */
/* Given a binary tree, we install cameras on the nodes
* of the tree. Each camera at a node can monitor
* its parent, itself, and its immediate children.
* Calculate the minimum number of cameras needed to
* monitor all nodes of the tree.
*************************************
* Apply a recusion function dfs.
* Return 0 if it's a leaf.
* Return 1 if it's a parent of a leaf, with a camera on this node.
* Return 2 if it's coverd, without a camera on this node.
* For each node,
* if it has a child, which is leaf (node 0), then it needs camera.
* if it has a child, which is the parent of a leaf (node 1), then it's covered.
* If it needs camera, then res++ and we return 1.
* If it's covered, we return 2.
* Otherwise, we return 0.*/
int resCamera = 0;
int minCameraCoverHelper(TreeNode* root) {
	if (!root) return 2;
	int left = minCameraCoverHelper(root->left);
	int right = minCameraCoverHelper(root->right);
	if (left == 0 || right == 0) {
		++resCamera;
		return 1; // IMPORTANT. 
	}
	return left == 1 || right == 1 ? 2 : 0;
}

int minCameraCover(TreeNode* root) {
	return (minCameraCoverHelper(root) < 1 ? 1 : 0) + resCamera;
}

/* 1145. Binary Tree Coloring Game */
int leftTree = 0, rightTree = 0, val = 0;
int count(TreeNode* node) {
	if (!node) return 0;
	int l = count(node->left), r = count(node->right);
	if (node->val == val) {
		leftTree = l, rightTree = r;
	}
	return l + r + 1;
}
bool btreeGameWinningMove(TreeNode* root, int n, int x) {
	val = x, n = count(root);
	return max(max(leftTree, rightTree), n - leftTree - rightTree - 1) > n / 2;
}

/* 979. Distribute Coins in Binary Tree */
/* Given the root of a binary tree with N nodes, each node
* in the tree has node.val coins, and there are N coins total.
* In one move, we may choose two adjacent nodes and move one
* coin from one node to another.
* (The move may be from parent to child, or from child to parent.)
* Return the number of moves required to make
* every node have exactly one coin.
* Logic: traverse childs first (post-order traversal), and return the ballance of coins. */
int distributeCoins(TreeNode* node, int& res) {
	if (!node) return 0;
	int left = distributeCoins(node->left, res);
	int right = distributeCoins(node->right, res);
	res += abs(left) + abs(right);
	return node->val + left + right - 1;
}

int distributeCoins(TreeNode* root) {
	int res = 0;
	distributeCoins(root, res);
	return res;
}

/* 803. Bricks Falling When Hit -- HARD */
/* We have a grid of 1s and 0s; the 1s in a cell represent bricks.  A brick will not drop 
* if and only if it is directly connected to the top of the grid, or at least one of its 
* (4-way) adjacent bricks will not drop. We will do some erasures sequentially. Each time 
* we want to do the erasure at the location (i, j), the brick (if it exists) on that 
* location will disappear, and then some other bricks may drop because of that erasure.
* Return an array representing the number of bricks that will drop after each erasure 
* in sequence. */
void hitBricks(vector<vector<int>>& grid, int i, int j, unordered_set<int>& keep) {
	int m = grid.size(), n = grid[0].size();
	if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != 1 || keep.count(i * n + j)) return;
	keep.insert(i * n + j);
	hitBricks(grid, i - 1, j, keep);
	hitBricks(grid, i + 1, j, keep);
	hitBricks(grid, i, j - 1, keep);
	hitBricks(grid, i, j + 1, keep);
}

vector<int> hitBricks(vector<vector<int>>& grid, vector<vector<int>>& hits) {
	int k = hits.size(), m = grid.size(), n = grid[0].size();
	vector<int> res(k, 0);
	unordered_set<int> keep;
	// Step 1: 先把要去掉的所有砖头都先去掉，这样我们遍历所有相连的砖头就是最终还剩下的砖头.
	for (auto a : hits) grid[a[0]][a[1]] -= 1; 
	// Step 2: 把不会掉落的砖头位置存入一个HashSet中.
	for (int i = 0; i < n; ++i) {
		if (grid[0][i] == 1) hitBricks(grid, 0, i, keep);
	}
	// Step 3: 从最后一个砖头开始往回加，每加一个砖头，我们就以这个砖头为起点，DFS遍历其周围相连的砖头. 
	// 统计出当前没有掉落的砖头数量，当去掉某个砖头后，我们可以统计当前还连着的砖头数量，
	// 二者做差值就是掉落的砖头数量.
	for (int i = k - 1; i >= 0; --i) {
		int preSize = keep.size(), x = hits[i][0], y = hits[i][1];
		if (++grid[x][y] != 1) continue; 
		if ((x - 1 >= 0 && keep.count((x - 1) * n + y)) ||
			(x + 1 < m && keep.count((x + 1) * n + y)) ||
			(y - 1 >= 0 && keep.count(x * n + y - 1)) ||
			(y + 1 < n && keep.count(x * n + y + 1)) ||
			x == 0) {
			hitBricks(grid, x, y, keep);
			res[i] = keep.size() - preSize - 1;
		}
	}
	return res;
}

/* 1136. Parallel Courses */
/* There are N courses, labelled from 1 to N.
* We are given relations[i] = [X, Y], representing a
* prerequisite relationship between course X and course Y:
* course X has to be studied before course Y.
* In one semester you can study any number of courses
* as long as you have studied all the prerequisites for
* the course you are studying. Return the minimum number
* of semesters needed to study all courses.
* If there is no way to study all the courses, return -1. */
bool minimumSemesters(vector<vector<int>>& g, vector<int>& visited, vector<int>& res, int i) {
	if (visited[i] == 1) return true;
	if (visited[i] == -1) return false;

	visited[i] = -1;
	for (auto a : g[i]) {
		if (!minimumSemesters(g, visited, res, a)) return false;
		res[i] = max(res[i], 1 + res[a]);
	}
	visited[i] = 1;
	return true;
}

// -1: visiting;  0: not visit;  1: visited
int minimumSemesters(int N, vector<vector<int>>& relations) {
	int m = relations.size(), n = relations[0].size();
	vector<int> res(N + 1, 1);
	vector<int> visited(N + 1, 0);
	vector<vector<int>> g(N + 1, vector<int>());

	for (auto a : relations) {
		g[a[0]].push_back(a[1]);
	}

	for (int i = 0; i <= N; ++i) {
		if (!minimumSemesters(g, visited, res, i)) return -1;
	}
	return *max_element(res.begin(), res.end());
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

/* 753. Cracking the Safe */
/* There is a box protected by a password.
* The password is a sequence of n digits where each digit
* can be one of the first k digits 0, 1, ..., k-1.
* While entering a password, the last n digits entered
* will automatically be matched against the correct password.
* For example, assuming the correct password is "345",
* if you type "012345", the box will open because the
* correct password matches the suffix of the entered password.
* Return any password of minimum length that is guaranteed
* to open the box at some point of entering it. */
string crackSafe(int n, int k) {
	string res(n, '0');
	set<string> st({ res });

	for (int i = 0; i < pow(k, n); ++i) {
		string pre = res.substr(res.size() - n + 1, n - 1);
		// IMPORTANT: start from "k - 1"
		for (int j = k - 1; j >= 0; --j) {
			string cur = pre + to_string(j);
			if (!st.count(cur)) {
				st.insert(cur);
				res += to_string(j);
				break;
			}
		}
	}
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

/* 841. Keys and Rooms -- SAME AS "GRAPH VALID TREE" */
/* There are N rooms and you start in room 0.
* Each room has a distinct number in 0, 1, 2, ..., N-1,
* and each room may have some keys to access the next room.
* Return true if and only if you can enter every room.
* Input: [[1],[2],[3],[]]. Output: true
* Explanation:
* We start in room 0, and pick up key 1.
* We then go to room 1, and pick up key 2.
* We then go to room 2, and pick up key 3.
* We then go to room 3.  Since we were able to go to
* every room, we return true. */
bool hasCycleRooms(vector<vector<int>>& rooms, vector<int>& visited, int cur) {
	if (visited[cur]) return true;
	visited[cur] = 1;
	for (auto a : rooms[cur]) {
		if (visited[a]) continue; // IMPORTANT
		if (hasCycleRooms(rooms, visited, a)) return true;
	}
	return false;
}

bool canVisitAllRooms(vector<vector<int>>& rooms) {
	int n = rooms.size();
	vector<int> visited(n, 0);
	if (hasCycleRooms(rooms, visited, 0)) return false;
	for (int i = 0; i < n; ++i) {
		if (!visited[i]) return false;
	}
	return true;
}

/* 333. Largest BST Subtree */
/* Given a binary tree, find the largest subtree which is a Binary Search Tree (BST), 
* where largest means subtree with largest number of nodes in it. */
int countBST(TreeNode* root) {
	if (!root) return 0; 
	return 1 + countBST(root->left) + countBST(root->right);
}

bool isBST(TreeNode* root, int mx, int mn) {
	if (!root) return true; 
	if (root->val >= mx || root->val <= mn) return false; 
	return isBST(root->left, root->val, mn) && isBST(root->right, mx, root->val);
}

int largestBSTSubtree(TreeNode* root) {
	if (!root) return 0; 
	if (isBST(root, INT_MAX, INT_MIN)) return countBST(root); 
	return max(largestBSTSubtree(root->left), largestBSTSubtree(root->right)); 
}

/* 329. Longest Increasing Path in a Matrix */
/* Given an integer matrix, find the length of the longest increasing path.
* From each cell, you can either move to four directions:
* left, right, up or down. You may NOT move diagonally or
* move outside of the boundary (i.e. wrap-around is not allowed). */
int longestIncreasingPath(vector<vector<int>>& matrix, int i, int j, vector<vector<int>>& dp) {
	if (matrix.empty() || matrix[0].empty()) return 0; 
	if (dp[i][j]) return dp[i][j]; 
	int res = 1; 
	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		if (x >= 0 && x < matrix.size() && y >= 0 && matrix[0].size() && matrix[x][y] >= 1 + matrix[i][j]) {
			res = max(res, 1 + longestIncreasingPath(matrix, x, y, dp));
		}
	}
	return dp[i][j] = res; 
}

int longestIncreasingPath(vector<vector<int>>& matrix) {
	int m = matrix.size(), n = matrix[0].size(), res = 0; 
	vector<vector<int>> dp(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			res = max(res, longestIncreasingPath(matrix, i, j, dp));
		}
	}
	return res;
}

/* 300. Longest Increasing Subsequence */
/* Given an unsorted array of integers, find the length
* of longest increasing subsequence.
* Example: Input: [10,9,2,5,3,7,101,18]. Output: 4.
* Follow up: Could you improve it to O(n log n) time complexity? */
int lengthOfLIS(vector<int>& nums) {
	int n = nums.size(), res = 0;
	vector<int> dp(n, 1);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < i; ++j) {
			if (nums[j] < nums[i]) dp[i] = max(dp[i], 1 + dp[j]);
		}
		res = max(res, dp[i]);
	}
	return res;
}

/* 827. Making A Large Island */
/* In a 2D grid of 0s and 1s, we change at most one 0 to a 1.
* After, what is the size of the largest island? (An island is a 4-directionally connected group of 1s).
* Input: [[1, 0], [0, 1]]. Output: 3. Explanation: Change one 0 to 1 and connect two 1s, 
* then we get an island with area = 3. 1 <= grid.length = grid[0].length <= 50. 0 <= grid[i][j] <= 1.*/
int getConnectArea(vector<vector<int>>& grid, int i, int j, int ix) {
	int area = 1, n = grid.size(); 
	grid[i][j] = ix; 
	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1]; 
		if (x >= 0 && x < n && y >= 0 && y < n && grid[x][y] == 1) {
			area += getConnectArea(grid, x, y, ix);
		}
	}
	return area; 
}

int largestIsland(vector<vector<int>>& grid) {
	int n = grid[0].size(), res = 0, ix = 2; // IMPORTANT: "ix = 2"
	unordered_map<int, int> m; // {index, area}

	// First step: mark each connected area with the same number
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == 1) {
				m[ix] = getConnectArea(grid, i, j, ix);
				res = max(res, m[ix++]);
			}
		}
	}
	// Second step: check each '0' cell, try to connect its four directions and save the largest area
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == 0) {
				unordered_set<int> visited; 
				int cur = 1; 

				for (auto dir : dirs) {
					int x = i + dir[0], y = j + dir[1]; 
					if (x >= 0 && x < n && y >= 0 && y < n) ix = grid[x][y];
					if (ix > 1 && !visited.count(ix)) {
						visited.insert(ix); 
						cur += m[ix]; 
					}
				}
				res = max(res, cur);
			}
		}
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

/* 947. Most Stones Removed with Same Row or Column */
/* On a 2D plane, we place stones at some integer coordinate points.  Each coordinate point may 
* have at most one stone. Now, a move consists of removing a stone that shares a column or row
* with another stone on the grid. What is the largest possible number of moves we can make?
* Input: stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]. Output: 5*/
void numberOfIslands(vector<vector<int>>& stones, set<string>& visited, int i, int j) {
	string s = to_string(i) + "_" + to_string(j);
	//if (visited.count(s)) return; 
	visited.insert(s);

	for (auto a : stones) {
		if ((i == a[0] || j == a[1]) && !visited.count(to_string(a[0]) + "_" + to_string(a[1]))) {
			numberOfIslands(stones, visited, a[0], a[1]);
		}
	}
}

int numberOfIslands(vector<vector<int>>& stones) {
	int res = 0;
	set<string> visited;
	for (auto a : stones) {
		if (!visited.count(to_string(a[0]) + "_" + to_string(a[1]))) {
			++res;
			numberOfIslands(stones, visited, a[0], a[1]);
		}
	}
	return res;
}

int removeStones(vector<vector<int>>& stones) {
	int n = stones.size();
	// res = total stone number - number of connected islands
	return n - numberOfIslands(stones);
}

/* 1254. Number of Closed Islands */
/* Given a 2D grid consists of 0s (land) and 1s (water). An island is a maximal 4-directionally connected
* group of 0s and a closed island is an island totally (all left, top, right, bottom) surrounded by 1s.
* Return the number of closed islands. */
void closedIsland(vector<vector<int>>& grid, int i, int j) {
	int m = grid.size(), n = grid[0].size();
	if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 1) return;
	grid[i][j] = 1;

	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		closedIsland(grid, x, y);
	}
}

int closedIsland(vector<vector<int>>& grid) {
	int res = 0, m = grid.size(), n = grid[0].size();
	// Step 1: mark all islands from edges to 1
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
				closedIsland(grid, i, j);
			}
		}
	}
	// Step 2: count the inner islands. 
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == 0) {
				++res;
				closedIsland(grid, i, j);
			}
		}
	}
	return res;
}

/* 323. Number of Connected Components in an Undirected Graph */
/* Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes),
* write a function to find the number of connected components in an undirected graph.
* Example: Input: n = 5 and edges = [[0, 1], [1, 2], [3, 4]], Output: 2. */
void countComponents(vector<vector<int>>& g, vector<int>& visited, int i) {
	if (visited[i]) return; 
	visited[i] = 1; 

	for (auto a : g[i]) {
		countComponents(g, visited, a);
	}
}

int countComponents(int n, vector<vector<int>>& edges) {
	int res = 0; 
	vector<vector<int>> g(n, vector<int>());
	vector<int> visited(n, 0);

	for (auto a : edges) {
		g[a[0]].push_back(a[1]);
		g[a[1]].push_back(a[0]);
	}
	for (int i = 0; i < n; ++i) {
		if (!visited[i]) {
			++res; 
			countComponents(g, visited, i);
		}
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

/* 305. Number of Islands II -- UNION FIND*/
/* A 2d grid map of m rows and n columns is initially filled with water. We may perform an addLand 
* operation which turns the water at position (row, col) into a land. Given a list of positions 
* to operate, count the number of islands after each addLand operation. An island is surrounded 
* by water and is formed by connecting adjacent lands horizontally or vertically. 
* You may assume all four edges of the grid are all surrounded by water. 
* Example: Input: m = 3, n = 3, positions = [[0,0], [0,1], [1,2], [2,1]]. Output: [1,1,2,3]. */
int findRoot(vector<int>& roots, int id) {
	return roots[id] == id ? id : findRoot(roots, roots[id]);
}

vector<int> numIslands2(int m, int n, vector<vector<int>>& positions) {
	vector<int> res; 
	int cnt = 0; 
	// 最好能够将每个陆地都标记出其属于哪个岛屿. 很适合使用联合查找 Union Find 来做.
	// 一般来说 root 数组都是使用一维数组，方便一些，那么这里就可以将二维数组 encode 为一维的，
	// 于是需要一个长度为 m*n 的一维数组来标记各个位置属于哪个岛屿. 假设每个位置都是一个单独岛屿.
	vector<int> roots(m * n, -1);
	
	for (auto a : positions) {
		int id = a[0] * n + a[1]; 
		// 若某个岛屿位置编码的 root 值不为 -1，说明这是一个重复出现的位置，不需要重新计算了. 
		if (roots[id] != -1) {
			res.push_back(cnt); 
			continue; 
		}
		// 否则将其岛屿编号设置为其坐标位置.
		++cnt; 
		roots[id] = id; 
		// 开始遍历其上下左右的位置.
		for (auto dir : dirs) {
			int x = a[0] + dir[0], y = a[1] + dir[1], new_id = x * n + y;
			// 遇到越界或者岛屿标号为 -1 的情况直接跳过.
			if (x < 0 || x >= m || y < 0 || y >= n || roots[new_id] == -1) continue; 
			// 因为当前这两个 land 是相邻的，它们是属于一个岛屿，所以其 getRoot 函数的
			// 返回值 suppose 应该是相等的，但是如果返回值不同，说明需要合并岛屿.
			int p = findRoot(roots, id), q = findRoot(roots, new_id);
			if (p != q) {
				roots[p] = q; 
				--cnt; 
			}
		}
		res.push_back(cnt); 
	}
	return res; 
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

/* 1020. Number of Enclaves */
/* Given a 2D array A, each cell is 0 (representing sea) or
* 1 (representing land). A move consists of walking from
* one land square 4-directionally to another land square,
* or off the boundary of the grid. Return the number of
* land squares in the grid for which we cannot walk off
* the boundary of the grid in any number of moves. */
void numEnclaves_dfs(vector<vector<int>>& A, int i, int j) {
	if (A[i][j] == 2) return;
	A[i][j] = 2;
	int m = A.size(), n = A[0].size();

	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		if (x >= 0 && x < m && y >= 0 && y < n && A[x][y] == 1) {
			numEnclaves_dfs(A, x, y);
		}
	}
}

int numEnclaves(vector<vector<int>>& A) {
	int m = A.size(), n = A[0].size(), res = 0;
	vector<vector<int>> visited(m, vector<int>(n, 0));
	// Step 1: change edged connected "1" into "2"
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
if (A[i][j] == 1) {
	numEnclaves_dfs(A, i, j);
}
			}
		}
	}
	// Step 2: count all connected area
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (A[i][j] == 1) {
				++res;
			}
		}
	}
	return res;
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

/* 756. Pyramid Transition Matrix */
/* We are stacking blocks to form a pyramid. Each block has a color which is a one letter string.
* We start with a bottom row of bottom, represented as a single string. We also start with a list
* of allowed triples allowed. Each allowed triple is represented as a string of length 3.
* Return true if we can build the pyramid all the way to the top, otherwise false.
* Input: bottom = "BCD", allowed = ["BCG", "CDE", "GEA", "FFF"]. Output: true. */
bool pyramidTransition(string bottom, string up, unordered_map<string, vector<char>>& m) {
	if (bottom.size() == 2 && up.size() == 1) return true;
	if (bottom.size() == up.size() + 1) return pyramidTransition(up, "", m);

	int pos = up.size();
	string cur = bottom.substr(pos, 2);
	if (m.count(cur)) {
		for (auto a : m[cur]) {
			if (pyramidTransition(bottom, up + string(1, a), m)) return true;
		}
	}
	return false;
}

bool pyramidTransition(string bottom, vector<string>& allowed) {
	unordered_map<string, vector<char>> m;
	for (auto a : allowed) {
		m[a.substr(0, 2)].push_back(a[2]);
	}
	return pyramidTransition(bottom, "", m);
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

/* 934. Shortest Bridge */
/* In a given 2D binary array A, there are two islands.  (An island is a 4-directionally connected group 
* of 1s not connected to any other 1s.) Now, we may change 0s to 1s so as to connect the two islands 
* together to form 1 island. Return the smallest number of 0s that must be flipped.  
* (It is guaranteed that the answer is at least 1.) */
int paint(vector<vector<int>>& A, int i, int j) {
	int m = A.size(), res = 0;
	if (i < 0 || i >= m || j < 0 || j >= m || A[i][j] != 1) return 0;
	A[i][j] = 2;
	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		res += paint(A, x, y);
	}
	return 1 + res;
}

bool expand(vector<vector<int>>& A, int i, int j, int cnt) {
	if (i < 0 || j < 0 || i == A.size() || j == A.size()) return false;
	if (A[i][j] == 0) A[i][j] = cnt + 1;
	return A[i][j] == 1;
}

int shortestBridge(vector<vector<int>>& A) {
	int m = A.size(), cnt = 2;
	// paint first 
	for (int i = 0, found = 0; !found && i < m; ++i) {
		for (int j = 0; !found && j < m; ++j) {
			found = paint(A, i, j);
		}
	}
	for (cnt = 2; ; ++cnt) {
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				if (A[i][j] == cnt && (expand(A, i - 1, j, cnt) || expand(A, i + 1, j, cnt) || expand(A, i, j - 1, cnt) || expand(A, i, j + 1, cnt))) {
					return cnt - 2;
				}
			}
		}
	}
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

/* 776. Split BST -- ??? */
/* Given a Binary Search Tree (BST) with root node root, and a target value V, split the tree into two subtrees
* where one subtree has nodes that are all smaller or equal to the target value, while the other subtree 
* has all nodes that are greater than the target value.  It's not necessarily the case that the tree 
* contains a node with value V. */
vector<TreeNode*> splitBST(TreeNode* root, int V) {
	vector<TreeNode*> res{ NULL, NULL };
	if (!root) return res;

	if (root->val > V) {
		res = splitBST(root->left, V);
		root->left = res[1];
		res[1] = root;
	}
	else {
		res = splitBST(root->right, V);
		root->right = res[0];
		res[0] = root;
	}

	return res;
}

/* 130. Surrounded Regions */
/* Given a 2D board containing 'X' and 'O' (the letter O),
* capture all regions surrounded by 'X'.
* A region is captured by flipping all 'O's into 'X's
* in that surrounded region. */
void solve(vector<vector<char>>& board, int i, int j) {
	int m = board.size(), n = board[0].size();
	if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != 'O') return;
	board[i][j] = '$';
	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		solve(board, x, y);
	}
}

void solve(vector<vector<char>>& board) {
	if (board.empty() || board[0].empty()) return;
	int m = board.size(), n = board[0].size();
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
				if (board[i][j] == 'O') solve(board, i, j);
			}
		}
	}
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (board[i][j] == 'O') board[i][j] = 'X';
			else if (board[i][j] == '$') board[i][j] = 'O';
		}
	}
}

/* 778. Swim in Rising Water */
/* On an N x N grid, each square grid[i][j] represents the elevation at that point (i,j).
* Now rain starts to fall. At time t, the depth of the water everywhere is t. 
* You can swim from a square to another 4-directionally adjacent square if and only if
* the elevation of both squares individually are at most t. You can swim infinite distance in zero time. 
* Of course, you must stay within the boundaries of the grid during your swim.
* You start at the top left square (0, 0). What is the least time until you can reach 
* the bottom right square (N-1, N-1)? */
void swimInWater(vector<vector<int>>& grid, int i, int j, int cur, vector<vector<int> >& dp) {
	int n = grid.size();
	if (i < 0 || i >= n || j < 0 || j >= n || max(cur, grid[i][j]) >= dp[i][j]) return;
	dp[i][j] = max(cur, grid[i][j]);
	for (auto dir : dirs) {
		swimInWater(grid, i + dir[0], j + dir[1], dp[i][j], dp);
	}
}

int swimInWater(vector<vector<int>>& grid) {
	int n = grid.size();
	vector<vector<int> > dp(n, vector<int>(n, INT_MAX));
	swimInWater(grid, 0, 0, grid[0][0], dp);
	return dp.back().back();
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

/* 494. Target Sum */
/* You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. 
* Now you have 2 symbols + and -. For each integer, you should choose one from + and - as its new symbol.
* Find out how many ways to assign symbols to make sum of integers equal to target S.
* Input: nums is [1, 1, 1, 1, 1], S is 3. Output: 5 */
void findTargetSumWays(vector<int>& nums, long long S, int idx, int& res) {
	if (idx == nums.size()) {
		if (S == 0) {
			++res;
		}
		return;
	}
	findTargetSumWays(nums, S - nums[idx], idx + 1, res);
	findTargetSumWays(nums, S + nums[idx], idx + 1, res);
}

int findTargetSumWays(vector<int>& nums, int S) {
	int res = 0;
	findTargetSumWays(nums, S, 0, res);
	return res;
}





// ===========================================================

// =================== 4. BFS PROBLEMS =======================
/* 102. Binary Tree Level Order Traversal */
vector<vector<int>> levelOrder(TreeNode* root) {
	vector<vector<int>> res; 
	if (!root) return res; 
	queue<TreeNode*> q{ {root} };
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

/* 107. Binary Tree Level Order Traversal II */
vector<vector<int>> levelOrderBottom(TreeNode* root) {
	vector<vector<int> > res;
	if (!root) return res;
	queue<TreeNode* > q;
	q.push(root);

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
	return vector<vector<int> >(res.rbegin(), res.rend());
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

/* 429. N-ary Tree Level Order Traversal */
/* Given an n-ary tree, return the level order traversal of its nodes' values. 
* (ie, from left to right, level by level). */
vector<vector<int>> levelOrderNDary(NaryTreeNode* root) {
	vector<vector<int>> res; 
	if (!root) return res; 
	queue<NaryTreeNode*> q{ {root} };

	while (!q.empty()) {
		int n = q.size(); 
		vector<int> ind; 

		for (int i = 0; i < n; ++i) {
			auto t = q.front(); q.pop(); 
			ind.push_back(t->val);
			for (auto a : t->children) {
				q.push(a);
			}
		}
		res.push_back(ind);
	}
	return res; 
}

/* 1161. Maximum Level Sum of a Binary Tree */
/* Given the root of a binary tree, the level of its root is 1, the level of its children is 2, and so on.
* Return the smallest level X such that the sum of all the values of nodes at level X is maximal. */
int maxLevelSum(TreeNode* root) {
	if (!root) return 0;
	int mx = INT_MIN, ix = 0;
	unordered_map<int, int> m;  // {sum, level}
	queue<TreeNode*> q{ { root } };

	while (!q.empty()) {
		int n = q.size(), sum = 0;
		for (int i = 0; i < n; ++i) {
			auto t = q.front(); q.pop();
			sum += t->val;
			if (t->left) q.push(t->left);
			if (t->right) q.push(t->right);
		}

		m[sum] = ++ix;
		mx = max(mx, sum);
	}
	for (auto it : m) {
		if (it.first == mx) return it.second;
	}
	return 0;
}

/* 958. Check Completeness of a Binary Tree */
bool isCompleteTree(TreeNode* root) {
	//if (!root) return true; 
	bool b = false;
	queue<TreeNode*> q{ { root } };

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		if (!t) b = true;
		else {
			if (b) return false;
			// IMPORTANT. NO "if" condition
			q.push(t->left);
			q.push(t->right);
		}
	}
	return true;
}

/* 542. 01 Matrix  */
/* Given a matrix consists of 0 and 1,
* find the distance of the NEAREST 0 for each cell.
* The distance between two adjacent cells is 1. */
vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
	if (matrix.empty() || matrix[0].empty()) return matrix;
	int m = matrix.size(), n = matrix[0].size();
	queue<pair<int, int>> q;

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			// IMPORTANT: save all '0' point as starting points
			if (matrix[i][j] == 0) {
				q.push({ i, j });
			}
			else {
				matrix[i][j] = INT_MAX;
			}
		}
	}

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		for (auto dir : dirs) {
			int x = t.first + dir[0], y = t.second + dir[1];
			if (x < 0 || x >= m || y < 0 || y >= n || matrix[x][y] <= matrix[t.first][t.second]) continue;
			matrix[x][y] = matrix[t.first][t.second] + 1;
			q.push({ x, y });
		}
	}
	return matrix;
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

// 815. Bus Routes -- HARD
/* We have a list of bus routes. Each routes[i] is a bus route that
* the i-th bus repeats forever. For example if routes[0] = [1, 5, 7],
* this means that the first bus (0-th indexed) travels in the
* sequence 1->5->7->1->5->7->1->... forever.
* We start at bus stop S (initially not on a bus), and we want to go
* to bus stop T. Travelling by buses only, what is the least number
* of buses we must take to reach our destination? Return -1 if
* it is not possible.
* Example: Input: routes = [[1, 2, 7], [3, 6, 7]], S = 1, T = 6
* Output: 2. */
int numBusesToDestination(vector<vector<int>>& routes, int S, int T) {
	unordered_map<int, unordered_set<int>> m;
	unordered_set<int> visited;
	queue<pair<int, int> > q;
	q.push({ S, 0 });

	for (int i = 0; i < routes.size(); ++i) {
		for (auto a : routes[i]) {
			m[a].insert(i);
		}
	}

	while (!q.empty()) {
		auto stop = q.front().first;
		auto cnt = q.front().second;
		q.pop();
		if (stop == T) return cnt;

		for (auto i : m[stop]) {
			for (auto a : routes[i]) {
				if (!visited.count(a)) {
					visited.insert(a);
					q.push({ a, cnt + 1 });
				}
			}
		}
	}
	return -1;
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

/* 742. Closest Leaf in a Binary Tree */
/* Given a binary tree where every node has a unique value, and a target key k, find the 
* value of the nearest leaf node to target k in the tree. Here, nearest to a leaf means the least 
* number of edges travelled on the binary tree to reach any leaf of the tree. Also, a node is 
* called a leaf if it has no children.*/
void buildTreeMap2(TreeNode* node, TreeNode* pre, unordered_map<TreeNode*, vector<TreeNode*>>& m, 
	TreeNode*& start, int k) {
	if (!node) return;
	if (node->val == k) start = node; 

	if (pre) {
		m[node].push_back(pre);
		m[pre].push_back(node);
	}
	buildTreeMap2(node->left, node, m, start, k);
	buildTreeMap2(node->right, node, m, start, k);
}

int findClosestLeaf(TreeNode* root, int k) {
	unordered_map<TreeNode*, vector<TreeNode*> > m;
	TreeNode* start = NULL; 
	buildTreeMap2(root, NULL, m, start, k);
	unordered_set<TreeNode*> visited{ { start } };
	queue<TreeNode*> q{ { start } };

	while (!q.empty()) {
		auto t = q.front(); q.pop(); 
		if (!t->left && !t->right) return t->val; 
		visited.insert(t); 

		for (auto a : m[t]) {
			if (visited.count(a)) continue; 
			q.push(a); 
		}
	}
	return -1;
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

/* 854. K-Similar Strings */
/* Strings A and B are K-similar (for some non-negative integer K)
* if we can swap the positions of two letters in A exactly K times
* so that the resulting string equals B. Given two anagrams A and B,
* return the SMALLEST K for which A and B are K-similar.
* Input: A = "abc", B = "bca". Output: 2.
* (1) 1 <= A.length == B.length <= 20
* (2) A and B contain only lowercase letters from the set
*     {'a', 'b', 'c', 'd', 'e', 'f'}. */
int kSimilarity(string A, string B) {
	if (A.size() != B.size()) return 0;
	int n = A.size();
	unordered_set<string> visited;
	queue<pair<string, int>> q{ { { A, 0 } } };
	unordered_map<int, vector<int>> m;
	for (int i = 0; i < n; ++i) m[B[i] - 'a'].push_back(i);

	while (!q.empty()) {
		auto s = q.front().first;
		auto d = q.front().second;
		q.pop();
		if (s == B) return d;

		int i = 0;
		for (; i < n; ++i) {
			if (s[i] != B[i]) break;
		}
		// IMPORTANT POINT.
		for (auto j : m[s[i] - 'a']) {
			if (s[j] != B[j]) {
				string t = s;
				t[i] = s[j];
				t[j] = s[i];
				q.push({ t, d + 1 });
			}
		}
	}
	return 0;
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

/* 774. Minimize Max Distance to Gas Station. */
/* On a horizontal number line, we have gas stations at positions
* stations[0], stations[1], ..., stations[N-1], where N = stations.length.
* Now, we add K more gas stations so that D, the maximum distance between adjacent gas stations, 
* is minimized. Return the smallest possible value of D.
* Example: Input: stations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], K = 9
* Output: 0.500000. Notes: stations.length will be an integer in range [10, 2000].
* stations[i] will be an integer in range [0, 10^8]. K will be an integer in range [1, 10^6]. */
double minmaxGasDist(vector<int>& stations, int K) {
	double left = 0, right = 1e8, n = stations.size(); 
	while (right - left > 1e-6) {
		double mid = left + (right - left) / 2; 
		int cnt = 0; 
		for (int i = 1; i < n; ++i) {
			cnt += (stations[i] - stations[i - 1]) / mid;
		}
		if (cnt > K) left = mid;
		else right = mid;
	}
	return left; 
}

/* 433. Minimum Genetic Mutation */
/* A gene string can be represented by an 8-character long string, with choices from "A", "C", "G", "T".
* Suppose we need to investigate about a mutation (mutation from "start" to "end"), 
* where ONE mutation is defined as ONE single character changed in the gene string.
* For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation. Also, there is a given gene "bank", 
* which records all the valid gene mutations. A gene must be in the bank to make it a valid gene string. 
* Example: start: "AACCGGTT", end:  "AAACGGTA". bank: ["AACCGGTA", "AACCGCTA", "AAACGGTA"]. return: 2. */
int minMutation(string start, string end, vector<string>& bank) {
	if (start.size() != 8 || end.size() != 8 || bank.empty()) return -1;
	unordered_set<string> dict(bank.begin(), bank.end());
	unordered_set<string> visited; // save visited strings
	queue<string> q;
	vector<char> v{ 'A', 'C', 'G', 'T' };
	int level = 0;

	while (!q.empty()) {
		int n = q.size();
		for (int i = 0; i < n; ++i) {
			string s = q.front(); q.pop();
			if (s == end) return level;

			for (int j = 0; j < s.size(); ++j) {
				char old = s[j];
				for (auto c : v) {
					s[j] = c;
					if (dict.count(s) && !visited.count(s)) {
						visited.insert(s);
						q.push(s);
					}
				}
				s[j] = old;
			}
		}
		++level;
	}
	return -1;
}

/* 310. Minimum Height Trees */
/* For an undirected graph with tree characteristics, we can choose
* any node as the root. The result graph is then a rooted tree.
* Among all possible rooted trees, those with minimum height
* are called minimum height trees (MHTs). Given such a graph,
* write a function to find all the MHTs and return a list of
* their root labels. */
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
	vector<int> res;
	if (edges.empty()) return res;
	vector<unordered_set<int>> g(n); // IMPORTANT. 
	for (auto a : edges) {
		g[a[0]].insert(a[1]);
		g[a[1]].insert(a[0]);
	}
	queue<int> q;
	for (int i = 0; i < n; ++i) {
		if (g[i].size() == 1) q.push(i);
	}

	while (n > 2) {
		int size = q.size();
		n -= q.size();

		for (int i = 0; i < size; ++i) {
			auto t = q.front(); q.pop();

			for (auto a : g[t]) {
				g[a].erase(t);
				if (g[a].size() == 1) q.push(a);
			}
		}
	}

	while (!q.empty()) {
		res.push_back(q.front());
		q.pop();
	}
	return res;
}

/* 1197. Minimum Knight Moves */
/* In an infinite chess board with coordinates from -infinity to +infinity,
* you have a knight at square [0, 0]. A knight has 8 possible moves it can make,
* as illustrated below. Each move is two squares in a cardinal direction,
* then one square in an orthogonal direction. Return the MINIMUM number of steps
* needed to move the knight to the square [x, y].  It is guaranteed the
* answer exists. */
int minKnightMoves(int x, int y) {
	if (x == 0 && y == 0) return 0;
	int res = 0;
	queue<pair<int, int>> q;
	q.push({ 0, 0 });
	unordered_map<int, unordered_map<int, int>> visited;
	visited[0][0] = 1;

	while (!q.empty()) {
		res++;
		int size = q.size();

		for (int i = 0; i < size; ++i) {
			auto r = q.front().first;
			auto c = q.front().second;
			q.pop();

			for (auto dir : dirs2) {
				int newr = r + dir[0], newc = c + dir[1];
				if (newr == x && newc == y) return res;
				// IMPORTANT. 
				if (!visited[newr][newc] && x * newr >= 0 && y * newc >= 0) {
					visited[newr][newc] = 1;
					q.push({ newr, newc });
				}
			}
		}
	}
	return -1;
}

/* 743. Network Delay Time -- Dijkstra's algorithm */
/* There are N network nodes, labelled 1 to N. Given times,
* a list of travel times as directed edges times[i] = (u, v, w),
* where u is the source node, v is the target node,
* and w is the time it takes for a signal to travel from source to target.
* Now, we send a signal from a certain node K. How long will it take
* for ALL nodes to receive the signal? If it is impossible, return -1.
* Input: times = [[2,1,1],[2,3,1],[3,4,1]], N = 4, K = 2. Output: 2. */
int networkDelayTime(vector<vector<int>>& times, int N, int K) {
	int res = 0;
	vector<int> dist(N + 1, INT_MAX);
	dist[K] = 0; // IMPORTATNT. 
	vector<vector<int>> g(N + 1, vector<int>(N + 1, -1));
	for (auto a : times) {
		g[a[0]][a[1]] = a[2];
	}

	queue<int> q{ { K } };
	while (!q.empty()) {
		// For each node, create a set to store visited nodes
		unordered_set<int> visited;
		int size = q.size();

		for (int k = 0; k < size; ++k) {
			auto i = q.front(); q.pop();
			// For each node in queue, consider all other unvisited neighbors
			// update the tentative distance from source to node 'j'. 
			for (int j = 1; j <= N; ++j) {
				if (g[i][j] != -1 && dist[j] > dist[i] + g[i][j]) {
					if (!visited.count(j)) {
						visited.insert(j);
						q.push(j);
					}
					dist[j] = dist[i] + g[i][j];
				}
			}
		}
	}
	for (int i = 1; i <= N; ++i) {
		res = max(res, dist[i]);
	}
	return res == INT_MAX ? -1 : res; // IMPORTANT.
}

int networkDelayTime2(vector<vector<int>>& times, int N, int K) {
	int res = 0;
	vector<vector<int>> g(N + 1, vector<int>(N + 1, -1));
	for (auto a : times) {
		g[a[0]][a[1]] = a[2];
	}
	vector<int> dists(N + 1, INT_MAX);
	dists[K] = 0; 

	queue<int> q{ {K} };
	while (!q.empty()) {
		unordered_set<int> visited; 
		int n = q.size(); 

		for (int k = 0; k < n; ++k) {
			auto i = q.front(); q.pop(); 

			for (int j = 1; j <= N; ++j) {
				if (g[i][j] != -1 && dists[j] > dists[i] + g[i][j]) {
					if (!visited.count(j)) {
						visited.insert(j); 
						q.push(j); 
					}
				}
				dists[j] = dists[i] + g[i][j]; 
			}
		}
	}
	for (int i = 1; i <= N; ++i) {
		res = max(res, dists[i]);
	}
	return res == INT_MAX ? -1 : res; 
}

/* 752. Open the Lock */
/* You have a lock in front of you with 4 circular wheels.
* Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'.
* The wheels can rotate freely and wrap around: for example we can turn
* '9' to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.
* The lock initially starts at '0000', a string representing the state of the 4 wheels.
* You are given a list of deadends dead ends, meaning if the lock displays
* any of these codes, the wheels of the lock will stop turning and you will
* be unable to open it. Given a target representing the value of the wheels
* that will unlock the lock, return the minimum total number of turns required
* to open the lock, or -1 if it is impossible.
* Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202".
* Output: 6. */
int openLock(vector<string>& deadends, string target) {
	if (target == "0000") return 0;
	unordered_set<string> locks(deadends.begin(), deadends.end());
	if (locks.count("0000")) return -1;
	unordered_set<string> visited{ {"0000"} };
	int res = 0; 
	queue<string> q{ {"0000"} };

	while (!q.empty()) {
		++res; 
		int n = q.size(); 
		for (int i = 0; i < n; ++i) {
			auto t = q.front(); q.pop(); 
			
			for (int j = 0; j < t.size(); ++j) {
				for (int k = -1; k < 2; ++k) {
					if (k == 0) continue;
					string s = t;
					s[j] = (t[j] - '0' + 10 + k) % 10 + '0';
					if (s == target) return res; 

					if (!locks.count(s) && !visited.count(s)) {
						visited.insert(s); 
						q.push(s); 
					}
				}
			}
		}
	}
	return -1; 
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
	queue<string> q{ {s} };
	unordered_set<string> visited{ {s} };
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

/* 317. Shortest Distance from All Buildings */
/* You want to build a house on an empty land which reaches
* all buildings in the shortest amount of distance.
* You can only move up, down, left and right.
* You are given a 2D grid of values 0, 1 or 2, where:
* Each 0 marks an empty land which you can pass by freely.
* Each 1 marks a building which you cannot pass through.
* Each 2 marks an obstacle which you cannot pass through. */
int shortestDistance(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size(), val = 0, res = INT_MAX;
	vector<vector<int> > sums = grid;
	vector<vector<int> > dirs{ { -1, 0 },{ 1, 0 },{ 0, -1 },{ 0, 1 } };
	queue<pair<int, int> > q;

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == 1) {
				// 对于每一个建筑的位置都进行一次全图的BFS遍历，
				// 每次都建立一个dist的距离场. 
				res = INT_MAX;
				vector<vector<int> > dists = grid;
				q.push({ i, j });

				while (!q.empty()) {
					auto t = q.front(); q.pop();
					for (auto dir : dirs) {
						int x = t.first + dir[0], y = t.second + dir[1];
						if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == val) {
							--grid[x][y];

							dists[x][y] = dists[t.first][t.second] + 1;
							// 真正的距离场累加在sum中.
							sums[x][y] += dists[x][y] - 1;
							res = min(res, sums[x][y]);

							q.push({ x, y });
						}
					}
				}
				--val;
			}
		}
	}
	return res == INT_MAX ? -1 : res;
}

/* 847. Shortest Path Visiting All Nodes ?????*/
/* An undirected, connected graph of N nodes
* (labeled 0, 1, 2, ..., N-1) is given as graph.
* graph.length = N, and j != i is in the list graph[i] exactly once,
* if and only if nodes i and j are connected. Return the length of
* the shortest path that visits every node. You may start and stop
* at any node, you may revisit nodes multiple times, and you may
* reuse edges.
* Input: [[1,2,3],[0],[0],[0]]. Output: 4
* Explanation: One possible path is [1,0,2,0,3]. */
int shortestPathLength(vector<vector<int>>& graph) {
	int n = graph.size(), target = 0, res = 0;
	unordered_set<string> visited;
	queue<pair<int, int>> q;
	for (int i = 0; i < n; ++i) {
		int mask = (1 << i);
		target |= mask;
		visited.insert(to_string(mask) + "-" + to_string(i));
		q.push({ mask, i });
	}
	while (!q.empty()) {
		for (int i = q.size(); i > 0; --i) {
			auto cur = q.front(); q.pop();
			if (cur.first == target) return res;

			for (int next : graph[cur.second]) {
				int path = cur.first | (1 << next);
				string str = to_string(path) + "-" + to_string(next);
				if (visited.count(str)) continue;
				visited.insert(str);
				q.push({ path, next });
			}
		}
		++res;
	}
	return -1;
}

/* 1091. Shortest Path in Binary Matrix */
/* In an N by N square grid, each cell is either empty (0) or blocked (1).
* A clear path from top-left to bottom-right has length k if and only if
* it is composed of cells C_1, C_2, ..., C_k such that:
* (1) Adjacent cells C_i and C_{i+1} are connected 8-directionally
* (ie., they are different and share an edge or corner)
* (2) C_1 is at location (0, 0) (ie. has value grid[0][0])
* (3) C_k is at location (N-1, N-1) (ie. has value grid[N-1][N-1])
* (4) If C_i is located at (r, c), then grid[r][c] is empty (ie. grid[r][c] == 0).
* Return the length of the shortest such clear path from top-left to bottom-right.
* If such a path does not exist, return -1. */
int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
	if (grid[0][0] == 1 || grid.back().back() == 1) return -1;
	int n = grid[0].size(), res = 0;
	queue<pair<int, int>> q{ { { 0, 0 } } };
	//vector<vector<int>> visited(n, vector<int>(n, 0));
	//visited[0][0] = 1; 
	grid[0][0] = 2;

	while (!q.empty()) {
		++res;
		int size = q.size();
		for (int i = 0; i < size; ++i) {
			auto t = q.front(); q.pop();
			// IMPORTANT. HERE CHECK FOR RETURN.
			if (t.first == n - 1 && t.second == n - 1) return res;

			for (auto dir : dirs3) {
				int x = t.first + dir[0], y = t.second + dir[1];
				if (x >= 0 && x < n && y >= 0 && y < n && grid[x][y] == 0) {
					grid[x][y] = 2;
					q.push({ x, y });
				}
			}
		}
	}
	return -1;
}

/* 864. Shortest Path to Get All Keys */
/* We are given a 2-dimensional grid.
* (1) "." is an empty cell,
* (2) "#" is a wall,
* (3) "@" is the starting point, ("a", "b", ...) are keys
* (4) ("A", "B", ...) are locks.
* We start at the starting point, and one move consists of walking
* one space in one of the 4 cardinal directions. We cannot walk
* outside the grid, or walk into a wall.  If we walk over a key,
* we pick it up.  We can't walk over a lock unless we have the
* corresponding key. For some 1 <= K <= 6, there is exactly one
* lowercase and one uppercase letter of the first K letters of
* the English alphabet in the grid. This means that there is
* exactly one key for each lock, and one lock for each key;
* and also that the letters used to represent the keys and locks
* were chosen in the same order as the English alphabet.
* Return the LOWEST number of moves to acquire all keys.
* If it's impossible, return -1.
* Input: ["@.a.#","###.#","b.A.B"]. Output: 8.
* NOTE: 1 <= grid.length <= 30.
* [i][j] contains only '.', '#', '@', 'a'-'f' and 'A'-'F'
* The number of keys is in [1, 6].
* Each key has a different letter and opens exactly one lock.*/
// 将钥匙编码成二进制数，对应位上的0和1表示该钥匙是存在，
// 比如二进制数 111111 就表示六把钥匙都有了
int shortestPathAllKeys(vector<string>& grid) {
	int m = grid.size(), n = grid[0].size();
	queue<pair<int, int>> q; // corrdinate and key count 
	unordered_set<string> visited;
	int key = 0, res = 0;

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == '@') {
				q.push({ i * n + j, 0 });
				visited.insert(to_string(i * n + j) + "_0");
			}
			if (grid[i][j] >= 'a' && grid[i][j] <= 'f') ++key;
		}
	}
	while (!q.empty()) {
		int size = q.size();
		for (int i = 0; i < size; ++i) {
			auto t = q.front().first, cnt = q.front().second;
			q.pop();
			if (cnt == (1 << key) - 1) return res;

			for (auto dir : dirs) {
				int x = t / n + dir[0], y = t % n + dir[1], cnt_t = cnt;
				if (x < 0 || x >= m || y < 0 || y >= n) continue;
				auto c = grid[x][y];

				if (c == '#') continue;
				if (c >= 'A' && c <= 'F' && ((cnt_t >> (c - 'A')) & 1) == 0) continue;
				if (c >= 'a' && c <= 'f') cnt_t |= 1 << (c - 'a');

				string s = to_string(x * n + y) + "_" + to_string(cnt_t);
				if (!visited.count(s)) {
					visited.insert(s);
					q.push({ x * n + y, cnt_t });
				}
			}
		}
		++res;
	}
	return -1;
}

/* 773. Sliding Puzzle */
/* On a 2x3 board, there are 5 tiles represented by the integers 1 through 5, and an empty square 
* represented by 0. A move consists of choosing 0 and a 4-directionally adjacent number and swapping it.
* The state of the board is solved if and only if the board is [[1,2,3],[4,5,0]].
* Given a puzzle board, return the least number of moves required so that the state of the board is solved.
* If it is impossible for the state of the board to be solved, return -1.
* Input: board = [[1,2,3],[4,0,5]]. Output: 1. */
int slidingPuzzle(vector<vector<int>>& board) {
	int res = 0, m = board.size(), n = board[0].size();
	vector<vector<int>> dirs{ { 1,3 },{ 0,2,4 },{ 1,5 },{ 0,4 },{ 1,3,5 },{ 2,4 } };
	string target = "123450", start("");

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			start += to_string(board[i][j]);
		}
	}
	queue<string> q{ { start } };
	unordered_set<string> visited{ start };
	while (!q.empty()) {
		int n = q.size();
		for (int i = 0; i < n; ++i) {
			auto t = q.front(); q.pop();
			if (t == target) return res;
			int idx = t.find("0");
			for (auto a : dirs[idx]) {
				string s = t;
				swap(s[a], s[idx]);
				if (visited.count(s)) continue;
				visited.insert(s);
				q.push(s);
			}
		}
		++res;
	}
	return -1;
}

/* 909. Snakes and Ladders */
/* On an N x N board, the numbers from 1 to N*N are written boustrophedonically starting 
* from the bottom left of the board, and alternating direction each row. 
* You start on square 1 of the board (which is always in the last row and first column). 
* Each move, starting from square x, consists of the following:
* You choose a destination square S with number x+1, x+2, x+3, x+4, x+5, or x+6, 
* provided this number is <= N*N. If S has a snake or ladder, you move to the destination of 
* that snake or ladder.  Otherwise, you move to S. A board square on row r and column c has 
* a "snake or ladder" if board[r][c] != -1.The destination of that snake or ladder is board[r][c].
* Return the least number of moves required to reach square N*N.  If it is not possible, return -1. */
vector<int> updatePos(int next, int n) {
	int x = (next - 1) / n;
	int y = (next - 1) % n;
	if (x % 2 == 1) {
		y = n - 1 - y;
	}
	x = n - 1 - x;
	return { x, y };
}

int snakesAndLadders(vector<vector<int>>& board) {
	int n = board.size(), target = n * n;
	unordered_map<int, int> m; // pos and step mapping
	m[1] = 0;
	queue<int> q;
	q.push(1);

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		if (t == target) return m[t];

		for (int i = 1; i <= 6; ++i) {
			int next = t + i;
			if (next > target) break;
			// Key function is to update the next position.
			auto new_pos = updatePos(next, n);
			int nx = new_pos[0], ny = new_pos[1];

			if (board[nx][ny] != -1) {
				next = board[nx][ny];
			}
			if (!m.count(next)) {
				m[next] = m[t] + 1;
				q.push(next);
			}
		}
	}
	return -1;
}

/* 490. The Maze */
/* There is a ball in a maze with empty spaces and walls.
* The ball can go through empty spaces by rolling up, down,
* left or right, but it won't stop rolling until hitting a wall.
* When the ball stops, it could choose the next direction.
* Given the ball's start position, the destination and the maze,
* determine whether the ball could stop at the destination. */
bool hasPath(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
	if (maze.empty() || maze[0].empty()) return true;
	int m = maze.size(), n = maze[0].size();
	vector<vector<bool>> visited(m, vector<bool>(n, false));
	queue<pair<int, int>> q;
	q.push({ start[0], start[1] });
	visited[start[0]][start[1]] = true;

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		if (t.first == destination[0] && t.second == destination[1])
			return true;
		for (auto dir : dirs) {
			int x = t.first, y = t.second;
			while (x >= 0 && x < m && y >= 0 && y < n && maze[x][y] == 0) {
				x += dir[0]; y += dir[1];
			}
			x -= dir[0]; y -= dir[1]; // IMPORTANT.
			if (!visited[x][y]) {
				visited[x][y] = true;
				q.push({ x, y });
			}
		}
	}
	return false;
}

/* 505. The Maze II */
/* Find the shortest distance for the ball to stop at the destination.*/
int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
	int m = maze.size(), n = maze[0].size();
	vector<vector<int>> dists(m, vector<int>(n, INT_MAX));
	vector<vector<int>> dirs{ { 0,-1 },{ -1,0 },{ 0,1 },{ 1,0 } };
	queue<pair<int, int>> q;
	q.push({ start[0], start[1] });
	dists[start[0]][start[1]] = 0;
	while (!q.empty()) {
		auto t = q.front(); q.pop();
		for (auto d : dirs) {
			int x = t.first, y = t.second, dist = dists[t.first][t.second];
			while (x >= 0 && x < m && y >= 0 && y < n && maze[x][y] == 0) {
				x += d[0];
				y += d[1];
				++dist;
			}
			x -= d[0];
			y -= d[1];
			--dist;
			if (dists[x][y] > dist) {
				dists[x][y] = dist; // RELAXIATION.
				if (x != destination[0] || y != destination[1])
					q.push({ x, y });
			}
		}
	}
	int res = dists[destination[0]][destination[1]];
	return (res == INT_MAX) ? -1 : res;
}

/* 286. Walls and Gates */
/* You are given a m x n 2D grid initialized with these three possible values.
* -1 - A wall or an obstacle. 0 - A gate. INF - Infinity means an empty room.
* We use the value 231 - 1 = 2147483647 to represent INF as you may assume
* that the distance to a gate is less than 2147483647.
* Fill each empty room with the distance to its nearest gate.
* If it is impossible to reach a gate, it should be filled with INF. */
void wallsAndGates(vector<vector<int>>& rooms) {
	if (rooms.empty() || rooms[0].empty()) return;
	int m = rooms.size(), n = rooms[0].size();
	queue<pair<int, int> > q;

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (rooms[i][j] == 0) q.push({ i, j });
		}
	}

	while (!q.empty()) {
		auto t = q.front(); q.pop();
		for (auto dir : dirs) {
			int x = t.first + dir[0], y = t.second + dir[1];

			if (x < 0 || x >= m || y < 0 || y >= n || rooms[x][y] <= rooms[t.first][t.second]) continue;
			rooms[x][y] = rooms[t.first][t.second] + 1;
			q.push({ x, y });
		}
	}
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

/* 407. Trapping Rain Water II */
int trapRainWater(vector<vector<int>>& heightMap) {
	if (heightMap.empty() || heightMap[0].empty()) return 0;

	int m = heightMap.size(), n = heightMap[0].size(), mx = 0, res = 0;
	// IMPORTANT.
	priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
	vector<vector<int>> visited(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			// START FROM EDGES.
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
				q.push({ heightMap[i][j], i * n + j });
				visited[i][j] = 1;
			}
		}
	}
	while (!q.empty()) {
		auto t = q.top(); q.pop();
		int h = t.first, i = t.second / n, j = t.second % n;
		mx = max(mx, h);

		for (auto dir : dirs) {
			int x = i + dir[0], y = j + dir[1];

			if (x < 0 || x >= m || y < 0 || y >= n || visited[x][y]) continue;
			visited[x][y] = 1;

			if (heightMap[x][y] < mx) res += mx - heightMap[x][y];
			q.push({ heightMap[x][y], x * n + y });
		}
	}

	return res;
}


// ===========================================================

// ================== 5. UNION FIND PROBLEMS =================
/* 803. Bricks Falling When Hit -- HARD */
/*
vector<int> hitBricksUionFind(vector<vector<int>>& grid, vector<vector<int>>& hits) {
}
*/

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

/* 305. Number of Islands II -- UNION FIND*/

/* 839. Similar String Groups */
/* Two strings X and Y are similar if we can swap two letters (in different positions) of X, 
* so that it equals Y. For example, "tars" and "rats" are similar (swapping at positions 0 and 2), 
* and "rats" and "arts" are similar, but "star" is not similar to "tars", "rats", or "arts".
* Together, these form two connected groups by similarity: {"tars", "rats", "arts"} and {"star"}.  
* Notice that "tars" and "arts" are in the same group even though they are not similar.  
* Formally, each group is such that a word is in the group if and only if it is similar to at least 
* one other word in the group. We are given a list A of strings.  Every string in A is an anagram of 
* every other string in A.  How many groups are there? Input: A = ["tars","rats","arts","star"]
* Output: 2. */
bool isSimilar(string& s, string& p) {
	if (s.size() != p.size()) return false;
	int cnt = 0;
	for (int i = 0; i < s.size(); ++i) {
		if (s[i] != p[i] && ++cnt > 2) return false;
	}
	return true;
}

string find(string s, unordered_map<string, string>& m) {
	return m[s] == s ? s : m[s] = find(m[s], m);
}

void numSimilarGroups(vector<string>& A, unordered_map<string, string>& m, int& res) {
	for (int i = 0; i < A.size(); ++i) {
		if (m.count(A[i])) continue;
		m[A[i]] = A[i];
		++res;

		for (int j = 0; j < i; ++j) {
			if (isSimilar(A[i], A[j])) {
				string x = find(A[i], m), y = find(A[j], m);
				if (x != y) {
					--res;
					m[x] = y;
				}
			}
		}
	}
}
int numSimilarGroups(vector<string>& A) {
	int res = 0;
	unordered_map<string, string> m;
	numSimilarGroups(A, m, res);
	return res;
}


// ===========================================================

// ================== 6. TOPO SORT PROBLEMS ==================
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



// ===========================================================

// =================== 7. BACKTRACKING PROBLEMS ==============
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

/* 40. Combination sum II */
/* Given a collection of candidate numbers (candidates) and a target number (target),
* find all unique combinations in candidates where the candidate numbers sums to target.
* Each number in candidates may only be used once in the combination. */
void combinationSum2(vector<int>& candidates, int target, int idx, vector<int>& ind, set<vector<int>>& res) {
	if (target < 0) return;
	if (target == 0 && !ind.empty()) {
		res.insert(ind);
		return;
	}
	for (int i = idx; i < candidates.size(); ++i) {
		ind.push_back(candidates[i]);
		combinationSum2(candidates, target - candidates[i], i + 1, ind, res); // DIFFERENCE
		ind.pop_back(); // BACKTRACKING
	}
}

vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
	set<vector<int>> res;
	vector<int> ind;
	sort(candidates.begin(), candidates.end());
	combinationSum2(candidates, target, 0, ind, res);
	return vector<vector<int>>(res.begin(), res.end());
}

/* 216. Combination sum III */
/* Find all possible combinations of k numbers that add up to a number n, given that only numbers
* from 1 to 9 can be used and each combination should be a unique set of numbers.*/
void combinationSum3(int k, int n, int idx, vector<int>& ind, vector<vector<int>>& res) {
	if (n < 0) return;
	if (ind.size() == k && n == 0) {
		res.push_back(ind);
		return;
	}

	for (int i = idx; i <= 9; ++i) { // here use "i = idx" to make sure each number only use once
		ind.push_back(i);
		combinationSum3(k, n - i, i + 1, ind, res);
		ind.pop_back(); // BACKTRACKING
	}
}

vector<vector<int>> combinationSum3(int k, int n) {
	vector<vector<int>> res;
	vector<int> ind;
	combinationSum3(k, n, 1, ind, res);
	return res;
}

// 254. Factor Combinations
void getFactors(int n, int idx, vector<int>& ind, vector<vector<int>>& res) {
	if (n == 1 && ind.size() > 1) { // IMPORTANT: "ind.size() > 1"
		res.push_back(ind);
		return;
	}
	for (int i = idx; i <= n; ++i) {
		if (n % i == 0) {
			ind.push_back(i);
			getFactors(n / i, i, ind, res);
			ind.pop_back(); // BACKTRACKING
		}
	}
}

vector<vector<int>> getFactors(int n) {
	vector<vector<int>> res;
	vector<int> ind;
	getFactors(n, 2, ind, res);
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

/* 526. beatuiful arrangement */
/* Suppose you have N integers from 1 to N. We define a beautiful arrangement as an array
* that is constructed by these N numbers successfully if one of the following is true
* for the ith position (1 <= i <= N) in this array:
* (1) The number at the ith position is divisible by i.
* (2) i is divisible by the number at the ith position. */
void countArrangement(int n, vector<int>& visited, int idx, int& res) {
	if (idx > n) {
		++res;
		return;
	}
	for (int i = 1; i <= n; ++i) {
		if (!visited[i] && (idx % i == 0 || i % idx == 0)) {
			visited[i] = 1;
			countArrangement(n, visited, idx + 1, res);
			visited[i] = 0; // BACKTRACKING
		}
	}
}

int countArrangement(int n) {
	int res = 0;
	vector<int> visited(n + 1, 0);
	countArrangement(n, visited, 1, res);
	return res;
}

// 401. Binary watch
void generateWatch(vector<int>& vec, int cnt, int idx, int ind, vector<int>& res) {
	if (cnt == 0) {
		res.push_back(ind);
		return;
	}
	for (int i = idx; i < vec.size(); ++i) {
		generateWatch(vec, cnt - 1, i + 1, ind + vec[i], res);
	}
}
// Generate possible numbers using only "cnt" number from "vec"
vector<int> generateWatch(vector<int>& vec, int cnt) {
	vector<int> res;
	generateWatch(vec, cnt, 0, 0, res);
	return res;
}

vector<string> readBinaryWatch(int n) {
	vector<string> res;
	vector<int> hour{ 1, 2, 4, 8 }, minute{ 1, 2, 4, 8, 16, 32 };

	for (int k = 1; k <= n; ++k) {
		vector<int> hours = generateWatch(hour, k);
		vector<int> minutes = generateWatch(minute, n - k);

		for (auto h : hours) {
			if (h > 11) continue;
			for (auto m : minutes) {
				if (m > 59) continue;
				res.push_back(to_string(h) + (m < 10 ? ":0" : ":") + to_string(m));
			}
		}
	}
	return res;
}

// binary watch 2. Use "bitset" to solve the problem
vector<string> readBinaryWatch2(int n) {
	vector<string> res;
	for (int h = 0; h < 12; ++h) {
		for (int m = 0; m < 60; ++m) {
			if (bitset<16>((h << 10) + m).count() == n) {
				res.push_back(to_string(h) + (m < 10 ? ":0" : ":") + to_string(m));
			}
		}
	}
	return res;
}

/* 1087. Brace Expansion */
/* Input: "{a,b}c{d,e}f". Output: ["acdf","acef","bcdf","bcef"] */
// select one char from each strings of "words" and save the new string. "k" is the number of strings
void expansion(vector<string>& words, int i, int k, string ind, vector<string>& res) {
	if (i == k) { // BACKTRACKING
		res.push_back(ind);
		return;
	}
	for (auto c : words[i]) {
		expansion(words, i + 1, k, ind + c, res);
	}
}

vector<string> expansion(string s) {
	vector<string> words(100);
	int idx = 0, flag = 0;
	// Step 1: generate strings from "s" and save in "words"
	for (auto c : s) {
		if (c == ',') continue;
		else if (c == '{') flag = 1;
		else if (c == '}') flag = 0, ++idx;
		else {
			words[idx] += c;
			if (!flag) ++idx;
		}
	}
	vector<string> res;
	// Step 2: sort each word in "words"
	for (int i = 0; i < idx; ++i) {
		sort(words[i].begin(), words[i].end());
	}
	// Step 3: do different combinations of chars of strings in "words"
	expansion(words, 0, idx, "", res);
	return res;
}

/* 1096. Brace Expansion II -- ??? */

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

/* 320. Generalized Abbreviation. String problem */
/* Input: "word"
* Output: ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2",
*          "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"] */
vector<string> generateAbbreviations(string word) {
	vector<string> res;
	int n = word.size();
	for (int i = 0; i < pow(2, n); ++i) {
		string ind("");
		int cnt = 0, t = i;
		for (int j = 0; j < n; ++j) {
			if (t & 1 == 1) {
				++cnt;
				if (j == n - 1) ind += to_string(cnt);
			}
			else {
				if (cnt) {
					ind += to_string(cnt);
					cnt = 0;
				}
				ind += word[j];
			}
			t >>= 1;
		}
		res.push_back(ind);
	}
	return res;
}

/* 411. Minimum Unique Word Abbreviation */
/* Given a target string and a set of strings in a dictionary, find an abbreviation of this
* target string with the smallest possible length such that it
* does not conflict with abbreviations of the strings in the dictionary. */
bool isAbbrevMatch(string abb, string s) {
	int m = abb.size(), n = s.size(), j = 0, cnt = 0;
	for (int i = 0; i < m; ++i) {
		if (abb[i] >= '0' && abb[i] <= '9') {
			if (cnt == 0 && abb[i] == '0') return false;
			cnt = cnt * 10 + (abb[i] - '0');
		}
		else {
			j += cnt;
			if (j > n || abb[i] != s[j++]) return false;
			cnt = 0;
		}
	}
	return j + cnt == n;
}
// Try "priority_queue" save the abbr and it's length (small to large length)
string minAbbreviation(string target, vector<string>& dictionary) {
	vector<string> strs = generateAbbreviations(target);
	sort(strs.begin(), strs.end(), [](string s, string p) {
		return s.size() < p.size() || (s.size() == p.size() && s < p);
	});

	for (auto s : strs) {
		bool noConflict = true;
		for (auto p : dictionary) {
			if (isAbbrevMatch(s, p)) {
				noConflict = false;
				break;
			}
		}
		if (noConflict) return s;
	}
	return "";
}

/* 89. Gray Code. Bit operator */
/* The gray code is a binary numeral system where two successive values differ in only one bit.
* Given a non-negative integer n representing the total number of bits in the code,
* print the sequence of gray code. A gray code sequence must begin with 0.    */
vector<int> grayCode(int n) {
	vector<int> res;
	for (int i = 0; i < pow(2, n); ++i) {
		res.push_back((i >> 1) ^ i);
	}
	return res;
}

/* 784. Letter Case Permutation */
/* Given a string S, we can transform every letter individually to be lowercase or uppercase to create another string.
* Return a list of all possible strings we could create.  */
void letterCasePermutation(string S, int idx, vector<string>& res) {
	if (idx == S.size()) { // BACKTRACKING
		res.push_back(S);
		return;
	}
	letterCasePermutation(S, idx + 1, res); // path 1 
	if (isalpha(S[idx])) {
		char c = S[idx];
		S[idx] = islower(c) ? toupper(c) : tolower(c);
		letterCasePermutation(S, idx + 1, res); // path 2
	}
}

vector<string> letterCasePermutation(string S) {
	vector<string> res;
	letterCasePermutation(S, 0, res);
	return res;
}

/* 51. N - Queens */
/* The n-queens puzzle is the problem of placing n queens on an n×n chessboard
* such that no two queens attack each other. */
bool isValidQueenMove(vector<int>& pos, int row, int col) {
	for (int i = 0; i < row; ++i) {
		if (pos[i] == col || abs(i - row) == abs(pos[i] - col)) return false;
	}
	return true;
}

void solveNQueens(vector<int>& pos, int row, vector<vector<string>>& res) {
	int n = pos.size(); 
	if (row == n) {
		vector<string> ind(n, string(n, '.'));
		for (int i = 0; i < n; ++i) {
			ind[i][pos[i]] = 'Q';
		}
		res.push_back(ind);
		return;
	}
	for (int col = 0; col < n; ++col) {
		if (isValidQueenMove(pos, row, col)) {
			pos[row] = col; 
			solveNQueens(pos, row + 1, res);
			pos[row] = -1; 
		}
	}
}

vector<vector<string>> solveNQueens(int n) {
	vector<vector<string>> res; 
	vector<int> pos(n, -1); 
	solveNQueens(pos, 0, res);
	return res; 
}

/* 131. Palindrome Partitioning
* Given a string s, partition s such that every substring of the partition is a palindrome.
* Return all possible palindrome partitioning of s. Input: "aab". Output: [["aa","b"],["a","a","b"]] */
bool isPalindrome(string s, int i, int j) {
	if (i > j) return false; 
	while (i <= j) {
		if (s[i++] != s[j--]) return false; 
	}
	return true;
}

void partition(string s, int ix, vector<string>& ind, vector<vector<string>>& res) {
	if (ix == s.size()) {
		res.push_back(ind);
		return;
	}
	for (int i = ix; i < s.size(); ++i) {
		if (isPalindrome(s, ix, i)) {
			ind.push_back(s.substr(ix, i - ix + 1));
			partition(s, i + 1, ind, res);
			ind.pop_back();
		}
	}
}

vector<vector<string>> partition(string s) {
	vector<vector<string> > res; 
	vector<string> ind; 
	partition(s, 0, ind, res);
	return res; 
}

/* 698. Partition to K Equal Sum Subsets */
/* Given an array of integers nums and a positive integer k, find whether it's possible to divide this array 
* into k non-empty subsets whose sums are all equal. Example 1: Input: nums = [4, 3, 2, 3, 5, 2, 1], k = 4
* Output: True. */
bool canPartitionKSubsets(vector<int>& nums, int k, int target, int start, int cursum, vector<int>& visited) {
	if (k == 1) return true; 
	if (cursum == target) return canPartitionKSubsets(nums, k - 1, target, 0, 0, visited);
	for (int i = start; i < nums.size(); ++i) {
		if (visited[i]) continue; 
		visited[i] = 1; 
		if (canPartitionKSubsets(nums, k, target, i + 1, cursum + nums[i], visited)) return true;
		visited[i] = 0; 
	}
	return false; 
}

bool canPartitionKSubsets(vector<int>& nums, int k) {
	int sum = accumulate(nums.begin(), nums.end(), 0);
	if (sum % k != 0) return false; 
	int target = sum / k, n = nums.size(); 
	vector<int> visited(n, 0);
	return canPartitionKSubsets(nums, k, target, 0, 0, visited);
}

/* 1219. Path with Maximum Gold */ 
/* In a gold mine grid of size m * n, each cell in this mine has an integer representing the amount of 
* gold in that cell, 0 if it is empty. Return the maximum amount of gold you can collect under the conditions:
* From your position you can walk one step to the left, right, up or down.
* You can't visit the same cell more than once.
* Never visit a cell with 0 gold.
* You can start and stop collecting gold from any position in the grid that has some gold. 
* Input: grid = [[0,6,0],[5,8,7],[0,9,0]]. Output: 24. Path to get the maximum gold, 9 -> 8 -> 7. */
int getMaximumGold(vector<vector<int>>& grid, vector<vector<int>>& visited, int i, int j) {
	int res = 0, m = grid.size(), n = grid[0].size();
	visited[i][j] = 1;

	for (auto dir : dirs) {
		int x = i + dir[0], y = j + dir[1];
		if (x >= 0 && x < m && y >= 0 && y < n && !visited[x][y] && grid[x][y] > 0) {
			res = max(res, grid[x][y] + getMaximumGold(grid, visited, x, y));
		}
	}
	visited[i][j] = 0; // BACKTRACKING.
	return res;
}

int getMaximumGold(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size(), res = 0;
	vector<vector<int>> visited(m, vector<int>(n, 0));

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] > 0) {
				res = max(res, grid[i][j] + getMaximumGold(grid, visited, i, j));
			}
		}
	}
	return res;
}

/* 638. Shopping Offers */
/* You are given the each item's price, a set of special offers, and the number we need to buy for each item. 
* The job is to output the lowest price you have to pay for exactly certain items as given, where you could 
* make optimal use of the special offers. Eg: Input: [2,5], [[3,0,5],[1,2,10]], [3,2]. Output: 14. */
int shoppingOffers(vector<int>& price, vector<vector<int>>& special, vector<int>& needs) {
	int n = price.size(), res = 0;
	for (int i = 0; i < n; ++i) res += price[i] * needs[i];

	for (auto s : special) {
		bool isValid = true;

		for (int i = 0; i < s.size() - 1; ++i) {
			if (s[i] > needs[i]) {
				isValid = false;
			}
			needs[i] -= s[i];
		}
		if (isValid) {
			res = min(res, shoppingOffers(price, special, needs) + s.back()); 
		}
		for (int i = 0; i < s.size() - 1; ++i) {
			needs[i] += s[i];
		}
	}
	return res;
}

// ===========================================================

// ==================== 8. LINKED LIST PROBLEMS ==============
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

/* 2. Add Two Numbers */
/* Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
* Output: 7 -> 0 -> 8 */
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
		if (l1) l1 = l1->next;
		if (l2) l2 = l2->next;
	}
	if (carry) pre->next = new ListNode(carry);
	return res->next;
}

/* 445. Add Two Numbers II */
/* Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
* Output: 7 -> 8 -> 0 -> 7  */
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
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

		res->val = sum % 10; // update "res" 
		ListNode* newnode = new ListNode(sum / 10);
		carry = sum / 10;

		newnode->next = res; // insert the "newnode" before "res"
		res = newnode;
	}
	if (carry) res->val = carry;
	return carry ? res : res->next;
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

/* 237. Delete Node in a Linked List */
void deleteNode(ListNode* node) {
	*node = *node->next;
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

/* 160. Intersection of Two Linked Lists */
int getLen(ListNode* head) {
	int res = 0;
	while (head) {
		++res;
		head = head->next;
	}
	return res;
}

/* 114. Flatten Binary Tree to Linked List */
/* Given a binary tree, flatten it to a linked list in-place. */
void flattenBTtoLL(TreeNode* root) {
	if (!root) return; 
	TreeNode* cur = root; 
	if (cur ->left) flattenBTtoLL(cur->left);
	if (cur ->right) flattenBTtoLL(cur->right);
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

/* 708. Insert into a Cyclic Sorted List */
/* Example: insert value = 6. */
ListNode* insert(ListNode* head, int insertVal) {
	if (!head) {
		ListNode* newhead = new ListNode(insertVal, nullptr);
		newhead->next = head;
		return newhead;
	}
	ListNode* pre = head, *cur = head->next;
	while (!(pre->val <= insertVal && insertVal <= cur->val) &&  // 5, 6, 7
		!(pre->val > cur->val && insertVal < cur->val) &&        // 8, 7, 6
		!(pre->val > cur->val && insertVal > pre->val)) {        // 6, 5, 4
		pre = pre->next;
		cur = cur->next;
		if (pre == head) break;
	}
	pre->next = new ListNode(insertVal, cur);
	return head;
}

/* 817. Linked List Components */
/* Input: head: 0->1->2->3, G = [0, 1, 3]. Output: 2 */
// solution 1
int numComponents(ListNode* head, vector<int>& G) {
	int res = 0;
	unordered_set<int> st(G.begin(), G.end());
	while (head) {
		if (st.count(head->val) && (!head->next || !st.count(head->next->val))) ++res;
		head = head->next;
	}
	return res;
}
// solution 2
int numComponents2(ListNode* head, vector<int>& G) {
	int res = 0; 
	unordered_set<int> st(G.begin(), G.end()); 
	while (head) {
		if (!st.count(head->val)) {
			head = head->next; 
			continue; 
		}
		++res;
		while (head && st.count(head->val)) {
			head = head->next; 
		}
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

/* 369. Plus One Linked List. Input: [1,2,3]. Output: [1,2,4] */
int plusOneHelper(ListNode* node) {
	if (!node) return 1; 
	int carry = plusOneHelper(node->next);
	int sum = carry + node->val; 
	node ->val = sum % 10; 
	return sum / 10; 
}

ListNode* plusOne(ListNode* head) {
	int carry = plusOneHelper(head);
	if (carry) {
		ListNode* newNode = new ListNode(1);
		newNode->next = head; 
		return newNode; 
	}
	return head; 
}

/* 143. Reorder List */
/* Given a singly linked list L: L0→L1→…→Ln-1→Ln, reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→… */
void mergeTwoLists2(ListNode* l1, ListNode* l2) {
	while (l1 && l2) {
		ListNode* t = l1->next;
		l1->next = l2;
		l2 = l2->next;
		l1->next->next = t;
		l1 = t;
	}
}

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

void reorderList(ListNode* head) {
	if (!head) return;
	ListNode* slow = head, *fast = head;
	while (fast->next && fast->next->next) {
		slow = slow->next;
		fast = fast->next->next;
	}
	fast = slow->next;
	slow->next = NULL;

	ListNode* newhead = reverseList(fast);
	mergeTwoLists2(head, newhead);
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

/* 725. Split Linked List in Parts */
/* Given a (singly) linked list with head node root, write a function to split the linked list 
* into k consecutive linked list "parts". The length of each part should be as equal as possible: 
* no two parts should have a size differing by more than 1. This may lead to some parts being null.
* Input: root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k = 3. Output: [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]. */
vector<ListNode*> splitListToParts(ListNode* root, int k) {
	vector<ListNode*> res(k);
	ListNode* cur = root; 
	int len = 0; 
	while (cur) {
		++len; 
		cur = cur->next; 
	}
	cur = root; 
	int avg = len / k, rem = len % k; 
	for (int i = 0; i < k && cur; ++i) {
		res[i] = cur; 
		// Trick: use "i < rem" to treat remaining nodes. 
		for (int j = 1; j < avg + (i < rem); ++j) {
			cur = cur->next; 
		}
		ListNode* t = cur->next; 
		cur->next = NULL; 
		cur = t; 
	}
	return res;
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

// ===========================================================

// ================== 9. BINARY SEARCH PROBLEMS ==============
/* 704. Binary Search */
/* Given a sorted (in ascending order) integer array nums of n elements and 
* a target value, write a function to search target in nums. If target exists, 
* then return its index, otherwise return -1. 
* Input: nums = [-1,0,3,5,9,12], target = 9. Output: 4. */
int search(vector<int>& nums, int target) {
	int n = nums.size(), l = 0, r = n - 1;
	while (l <= r) {
		int m = l + (r - l) / 2;
		if (nums[m] == target) return m;
		else if (nums[m] < target) l = m + 1;
		else r = m - 1;
	}
	return -1;
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

/* 69. Sqrt(x) （第一类拓展）
* Input: 8. Output: 2. */
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

/* 15. 3Sum */
/* Given an array nums of n integers, are there elements a, b, c in nums such that
 * a + b + c = 0? Find all unique triplets in the array which gives the sum of zero. 
 * Given array nums = [-1, 0, 1, 2, -1, -4] => [[-1, 0, 1], [-1, -1, 2]]. */
vector<vector<int>> threeSum(vector<int>& nums) {
	vector<vector<int>> res; 
	if (nums.empty()) return res; 
	sort(nums.begin(), nums.end()); 
	int n = nums.size(); 

	for (int i = 0; i < n; ++i) {
		if (nums[i] > 0) break; 
		if (i > 0 && nums[i] == nums[i - 1]) continue; 

		int target = -1 * nums[i], left = i + 1, right = n - 1; 
		while (left < right) {
			if (nums[left] + nums[right] == target) {
				res.push_back({ nums[i], nums[left], nums[right] });
				while (left < right && nums[left] == nums[left + 1]) ++left; 
				while (left < right && nums[right] == nums[right - 1]) --right;
				++left, --right;
			}
			else if (nums[left] + nums[right] < target) {
				++left; 
			}
			else {
				--right;
			}
		}
	}
	return res; 
}

/* 16. 3Sum Closest */
/* Given an array nums of n integers and an integer target, find three integers 
 * in nums such that the sum is closest to target. Return the sum of the three
 * integers. You may assume that each input would have exactly one solution.
 * Example: Given array nums = [-1, 2, 1, -4], and target = 1. => 2.  */
int threeSumClosest(vector<int>& nums, int target) {
	sort(nums.begin(), nums.end());
	int n = nums.size(), sum = nums[0] + nums[1] + nums[2];
	for (int i = 0; i < n - 1; ++i) {
		int diff = abs(target - sum), l = i + 1, r = n - 1;

		while (l < r) {
			int newsum = nums[i] + nums[l] + nums[r];
			int newdiff = abs(target - newsum);

			if (newsum < target) ++l;
			else --r;

			if (newdiff < diff) {
				sum = newsum;
				diff = newdiff;
			}
		}
	}
	return sum;
}

/* 259. 3Sum Smaller */
/* Given an array of n integers nums and a target, find the number of index triplets 
 * i, j, k with 0 <= i < j < k < n that satisfy the condition 
 * nums[i] + nums[j] + nums[k] < target. Input: nums = [-2,0,1,3], and target = 2. => 2. */
int threeSumSmaller(vector<int>& nums, int target) {
	sort(nums.begin(), nums.end());
	int res = 0, n = nums.size();

	for (int i = 0; i < n - 2; ++i) {
		int l = i + 1, r = n - 1, t = target - nums[i];

		while (l < r) {
			if (nums[l] + nums[r] < t) {
				res += r - l;
				++l;
			}
			else {
				--r;
			}
		}
	}
	return res;
}

/* 18. 4Sum */
/* Given an array nums of n integers and an integer target, are there elements 
* a, b, c, and d in nums such that a + b + c + d = target? Find all unique 
* quadruplets in the array which gives the sum of target. */
vector<vector<int>> fourSum(vector<int>& nums, int target) {
	sort(nums.begin(), nums.end());
	int n = nums.size();
	set<vector<int>> res;

	for (int i = 0; i < n - 3; ++i) {
		for (int j = i + 1; j < n - 2; ++j) {
			int t = target - (nums[i] + nums[j]), l = j + 1, r = n - 1;
			while (l < r) {
				if (nums[l] + nums[r] == t) {
					res.insert({ nums[i], nums[j], nums[l], nums[r] });
					while (l < r && nums[l] == nums[l + 1]) ++l;
					while (l < r && nums[r] == nums[r - 1]) --r;
					++l, --r;
				}
				else if (nums[l] + nums[r] < t) {
					++l;
				}
				else {
					--r;
				}
			}
		}
	}
	return vector<vector<int>>(res.begin(), res.end());
}

/* 454. 4Sum II */
/* Given four lists A, B, C, D of integer values, compute how many
* tuples (i, j, k, l) there are such that A[i] + B[j] + C[k] + D[l] is zero.
* To make problem a bit easier, all A, B, C, D have same length of N
* where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1
* and the result is guaranteed to be at most 231 - 1. */
int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
	unordered_map<int, int> m;
	int n = A.size(), res = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			++m[A[i] + B[j]];
		}
	}
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			int t = -1 * (C[i] + D[j]);
			if (m.count(t)) res += m[t];
		}
	}
	return res;
}

/* 1011. Capacity To Ship Packages Within D Days (第四类） */
/* A conveyor belt has packages that must be shipped from one port
* to another within D days. The i-th package on the conveyor belt
* has a weight of weights[i]. Each day, we load the ship with
* packages on the conveyor belt (in the order given by weights).
* We may not load more weight than the maximum weight capacity of the ship.
* Return the LEAST weight capacity of the ship that will result in
* all the packages on the conveyor belt being shipped within D days.
* Input: weights = [1,2,3,4,5,6,7,8,9,10], D = 5. Output: 15.
* 1 <= D <= weights.length <= 50000. 1 <= weights[i] <= 500.
* ************************ RUNNING OUTPUT ***********************:
* l = 10, r = 55, m = 32
* l = 10, r = 32, m = 21
* l = 10, r = 21, m = 15
* l = 10, r = 15, m = 12
* l = 13, r = 15, m = 14.
* ***************************************************************/
// PRETTY TRICKY PROBLEM.  -- TRIPPLE THINK!!!
int shipWithinDays(vector<int>& weights, int D) {
	int l = *max_element(weights.begin(), weights.end());
	int r = accumulate(weights.begin(), weights.end(), 0);

	while (l < r) {
		int m = l + (r - l) / 2;
		// IMPORTANT. "cur = 0" 
		int cur = 0, need = 1;
		// IMPORTANT. " cur += weights[i++]"
		for (int i = 0; i < weights.size() && need <= D; cur += weights[i++]) {
			if (cur + weights[i] > m) {
				cur = 0;
				++need;
			}
		}
		if (need > D) l = m + 1;
		else r = m;
	}
	return l;
}

/* 1231. Divide Chocolate -- HARD */
/* You have one chocolate bar that consists of some chunks. Each chunk has its
* own sweetness given by the array sweetness. You want to share the chocolate
* with your K friends so you start cutting the chocolate bar into K+1 pieces
* using K cuts, each piece consists of some consecutive chunks. Being generous,
* you will eat the piece with the minimum total sweetness and give the other
* pieces to your friends. Find the maximum total sweetness of the piece you can
* get by cutting the chocolate bar optimally.
* 0 <= K < sweetness.length <= 10^4. 1 <= sweetness[i] <= 10^5.
* Input: sweetness = [1,2,3,4,5,6,7,8,9], K = 5. Output: 6.
* Input: sweetness = [5,6,7,8,9,1,2,3,4], K = 8. Output: 1.
* Input: sweetness = [1,2,2,1,2,2,1,2,2], K = 2. Output: 5. */
int maximizeSweetness(vector<int>& sweetness, int K) {
	int l = 1, r = 1e9 / (K + 1);

	while (l < r) {
		int m = (l + r + 1) / 2;
		int cur = 0, need = 0;
		for (auto a : sweetness) {
			if ((cur += a) >= m) {
				cur = 0;
				if (++need > K) break;
			}
		}
		if (need > K) l = m;
		else r = m - 1;
	}
	return l;
}

/* 315. Count of Smaller Numbers After Self -- HARD （第五类）*/ 
/* Given an integer array nums and you have to return a new counts array.
* The counts array has the property where counts[i] is the number of
* smaller elements to the right of nums[i].
* Input: [5,2,6,1]. Output: [2,1,1,0]. */
vector<int> countSmaller(vector<int>& nums) {
	int n = nums.size();
	vector<int> v, res(n, -1);

	for (int i = n - 1; i >= 0; --i) {
		int left = 0, right = v.size();
		// IMPORTANT.
		while (left < right) {
			int mid = left + (right - left) / 2;
			// IMPORTANT. "=" treatment
			// IF "SMALLER AND EQUAL" THEN PUT "=" 
			// INTO "LEFT"
			if (v[mid] < nums[i]) left = mid + 1;
			else right = mid;
		}
		v.insert(v.begin() + left, nums[i]);
		res[i] = right;
	}
	return res;
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

/* 287. Find the Duplicate Number （第四类）
* Given an array nums containing n + 1 integers where each integer
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

/* 475. Heaters
* your input will be the positions of houses and heaters seperately,
* and your expected output will be the minimum radius
* standard of heaters. Input: [1,2,3,4],[1,4]. Output: 1. */
int findRadius(vector<int>& houses, vector<int>& heaters) {
	sort(houses.begin(), houses.end());
	sort(heaters.begin(), heaters.end());

	int m = houses.size(), n = heaters.size(), res = 0;
	for (int i = 0, j = 0; i < m; ++i) {
		int cur = houses[i];
		while (j < n - 1 && abs(cur - heaters[j]) >= abs(cur - heaters[j + 1]))
			++j;
		res = max(res, abs(cur - heaters[j]));
	}
	return res;
}

/* 683. K Empty Slots */
/* Given an integer K, find out the minimum day number such that there exists two turned on bulbs 
* that have exactly K bulbs between them that are all turned off. */
int kEmptySlots(vector<int>& flowers, int K) {
	int res = INT_MAX, left = 0, right = K + 1, n = flowers.size();
	vector<int> days(n, 0);
	for (int i = 0; i < n; ++i) days[flowers[i] - 1] = i + 1;
	for (int i = 0; right < n; ++i) {
		if (days[i] < days[left] || days[i] <= days[right]) {
			if (i == right) res = min(res, max(days[left], days[right]));
			left = i;
			right = K + 1 + i;
		}
	}
	return (res == INT_MAX) ? -1 : res;
}

/* 230. Kth Smallest Element in a BST */
/* Given a binary search tree, write a function
* kthSmallest to find the kth smallest element in it. */
int kthSmallest(TreeNode* root, int k) {
	int cnt = 0;
	stack<TreeNode*> st;
	TreeNode* p = root;

	while (p || !st.empty()) {
		// IMPORTANT. "while (p)"
		while (p) {
			st.push(p);
			p = p->left;
		}
		p = st.top(); st.pop();
		++cnt;
		if (cnt == k) return p->val;
		p = p->right;
	}
	return -1;
}

/* 378. Kth Smallest Element in a Sorted Matrix （第四类） */
/* Given a n x n matrix where each of the rows and columns
* are sorted in ascending order, find the kth smallest
* element in the matrix. Note that it is the kth smallest
* element in the sorted order, not the kth distinct element.
* matrix = [[ 1,  5,  9], [10, 11, 13], [12, 13, 15]],  k = 8, return 13. */
int kthSmallest(vector<vector<int>>& matrix, int k) {
	int left = matrix[0][0], right = matrix.back().back();

	while (left < right) {
		int mid = left + (right - left) / 2;
		int cnt = 0;
		// IMPORTANT. COUNT HOW MANY THAT IS SMALLER THAN SOME NUMBER.
		for (int i = 0; i < matrix.size(); ++i) {
			cnt += upper_bound(matrix[i].begin(), matrix[i].end(), mid) - matrix[i].begin();
		}
		if (cnt < k) left = mid + 1;
		else right = mid;
	}
	return right;
}

/* 668. Kth Smallest Number in Multiplication Table （第四类）*/
/* Nearly every one have used the Multiplication Table.
* But could you find out the k-th smallest number quickly
* from the multiplication table? Given the height m and
* the length n of a m * n Multiplication Table, and a
* positive integer k, you need to return the k-th smallest
* number in this table.
* Input: m = 3, n = 3, k = 5. Output:3.
* Explanation: The Multiplication Table:
* 1	2	3
* 2	4	6
* 3	6	9. */
int findKthNumber(int m, int n, int k) {
	int left = 1, right = m * n;

	while (left < right) {
		int mid = left + (right - left) / 2;
		int cnt = 0;

		for (int i = 1; i <= m; ++i) {
			if (mid > i * n) cnt += n;
			else cnt += mid / i;
		}
		if (cnt < k) left = mid + 1;
		else right = mid;
	}
	return right;
}

/* 875. Koko Eating Bananas */
/* Koko loves to eat bananas. There are N piles of bananas,
* the i-th pile has piles[i] bananas. The guards have gone and
* will come back in H hours. Koko can decide her bananas-per-hour
* eating speed of K.  Each hour, she chooses some pile of bananas,
* and eats K bananas from that pile.  If the pile has less than
* K bananas, she eats all of them instead, and won't eat any
* more bananas during this hour. Koko likes to eat slowly,
* but still wants to finish eating all the bananas before the
* guards come back. Return the minimum integer K such that she
* can eat all the bananas within H hours.
* NOTES: 1 <= piles.length <= 10^4;
*        piles.length <= H <= 10^9;
*        1 <= piles[i] <= 10^9.
* Input: piles = [3,6,7,11], H = 8. Output: 4.
* Input: piles = [30,11,23,4,20], H = 5. Output: 30.
* Input: piles = [30,11,23,4,20], H = 6. Output: 23. */
/* Solution 1:
* Each hour, Koko chooses some pile of bananas, and eats K bananas from
* that pile. There is a limited range of K's to enable her to eat
* all the bananas within H hours. We ought to reduce the searching
* space and to return the minimum valid K. */
/*
int minEatingSpeed(vector<int>& piles, int H) {
}
*/

// Solution 2: NOT THAT INITUITIVE
bool canEatAll(vector<int>& piles, int n, int h) {
	int cnt = 0;
	for (auto a : piles) {
		cnt += a / n;
		if (a % n != 0) cnt++;
	}
	return cnt <= h;
}

int minEatingSpeed2(vector<int>& piles, int H) {
	int left = 1, right = 1e9;

	while (left <= right) {
		int mid = left + (right - left) / 2;
		if (canEatAll(piles, mid, H)) {
			right = mid - 1;
		}
		else {
			left = mid + 1;
		}
	}
	return left;
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

/* 374. Guess Number Higher or Lower
int guessNumber(int n) {
if (guess(n) == 0) return n;
int left = 1, right = n;
while (left <= right) {
int mid = left + (right - left) / 2;
if (guess(mid) == 0) return mid;
else if (guess(mid) == 1) left = mid + 1;
else right = mid - 1;
}
return left;
}*/

// ===========================================================

// ============== 10. SEGAMENT TREE PROBLEMS =================
/* 307. Range Sum Query - Mutable */
/* Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
* The update(i, val) function modifies nums by updating the element at index i to val. 
* Given nums = [1, 3, 5]. sumRange(0, 2) -> 9. update(1, 2). sumRange(0, 2) -> 8. */
class NumArray {
public:
	NumArray(vector<int>& nums) {
		int n = nums.size(); 
		vec.resize(n, 0); 
		bit.resize(n + 1, 0);
	}

	void update(int i, int val) {
		int diff = val - vec[i];
		for (int j = i + 1; j < bit.size(); j += (j & (-j))) {
			bit[j] += diff; 
		}
		vec[i] = val; 
	}

	int sumRange(int i, int j) {
		return getSum(j + 1) - getSum(i);
	}

	int getSum(int i) {
		int res = 0; 
		for (int j = i; j > 0; j -= (j & -j)) {
			res += bit[j];
		}
		return res; 
	}

private:
	vector<int> vec, bit; 
};

/* 308. Range Sum Query 2D - Mutable */
/* Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined
 * by its upper left corner (row1, col1) and lower right corner (row2, col2). */
class NumMatrix {
public:
	NumMatrix(vector<vector<int>> matrix) {
		if (matrix.empty() || matrix[0].empty()) return;
		mat.resize(matrix.size() + 1, vector<int>(matrix[0].size() + 1, 0));
		bit.resize(matrix.size() + 1, vector<int>(matrix[0].size() + 1, 0));

		for (int i = 0; i < matrix.size(); ++i) {
			for (int j = 0; j < matrix[0].size(); ++j) {
				update(i, j, matrix[i][j]);
			}
		}
	}

	void update(int row, int col, int val) {
		int diff = val - mat[row + 1][col + 1];
		for (int i = row + 1; i < mat.size(); i += (i & -i)) {
			for (int j = col + 1; j < mat[0].size(); j += (j & -j)) {
				bit[i][j] += diff;
			}
		}
		mat[row + 1][col + 1] = val;
	}

	int sumRegion(int row1, int col1, int row2, int col2) {
		return getSum(row2 + 1, col2 + 1) - getSum(row2 + 1, col1) - getSum(row1, col2 + 1) + getSum(row1, col1);
	}

	int getSum(int row, int col) {
		int res = 0;
		for (int i = row; i > 0; i -= (i & -i)) {
			for (int j = col; j > 0; j -= (j & -j)) {
				res += bit[i][j];
			}
		}
		return res;
	}

private:
	vector<vector<int> > mat;
	vector<vector<int> > bit;
};



// ===========================================================

// =============== 11. BIT MANIPULATION PROBLEMS =============
/* 320. Generalized Abbreviation. String problem */
/* Input: "word"
* Output: ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2",
*          "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"] */
vector<string> generateAbbreviations(string word) {
	vector<string> res;
	int n = word.size();
	for (int i = 0; i < pow(2, n); ++i) {
		string ind("");
		int cnt = 0, t = i;
		for (int j = 0; j < n; ++j) {
			if (t & 1 == 1) {
				++cnt;
				if (j == n - 1) ind += to_string(cnt);
			}
			else {
				if (cnt) {
					ind += to_string(cnt);
					cnt = 0;
				}
				ind += word[j];
			}
			t >>= 1;
		}
		res.push_back(ind);
	}
	return res;
}

/* 318. Maximum Product of Word Lengths */
/* Given a string array words, find the maximum value of length(word[i]) * length(word[j]) 
* where the two words do not share common letters. You may assume that each word will contain 
* only lower case letters. If no such two words exist, return 0.
* Input: ["abcw","baz","foo","bar","xtfn","abcdef"]. Output: 16. 
* Explanation: The two words can be "abcw", "xtfn".*/
int maxProduct(vector<string>& words) {
	int n = words.size(), res = 0;;
	vector<int> masks(n, 0);
	for (int i = 0; i < n; ++i) {
		auto s = words[i]; 
		for (auto c : s) {
			masks[i] |= 1 << (c - 'a');
		}
		for (int j = 0; j < i; ++j) {
			if (!(masks[i] & masks[j])) {
				res = max(res, (int)(words[i].size() * words[j].size()));
			}
		}
	}
	return res; 
}

/* 421. Maximum XOR of Two Numbers in an Array */
/* Given a non-empty array of numbers, a0, a1, a2, … , an-1, where 0 ≤ ai < 231.
* Find the maximum result of ai XOR aj, where 0 ≤ i, j < n. Could you do this in O(n) runtime? */
int findMaximumXOR(vector<int>& nums) {
	int res = 0, n = nums.size();
	vector<int> masks(n, 0);

	for (int i = 0; i < n; ++i) {
		string s = to_string(nums[i]);
		for (auto c : s) {
			masks[i] |= (c - '0');
		}

		for (int j = 0; j < i; ++j) {
			if (!(masks[i] & masks[j])) {
				res = max(res, nums[i] ^ nums[j]);
			}
		}
	}
	return res;
}

/* 191. Number of 1 Bits */
/* Write a function that takes an unsigned integer and return the number of '1' bits it has 
* (also known as the Hamming weight). Input: 00000000000000000000000000001011. Output: 3. */
int hammingWeight(uint32_t n) {
	int res = 0; 
	for (int i = 31; i >= 0; --i) {
		if (n & 1 == 1) ++res; 
		n >>= 1; 
	}
	return res; 
}

/* 540. Single Element in a Sorted Array */
/* You are given a sorted array consisting of only integers where every element appears exactly twice,
* except for one element which appears exactly once. Find this single element that appears only once. */
int singleNonDuplicate(vector<int>& nums) {
	int res = 0;
	for (auto a : nums) {
		res ^= a;
	}
	return res;
}

/* 136. Single Number */
/* Given a non-empty array of integers, every element appears twice except for one. Find that single one. */
int singleNumber(vector<int>& nums) {
	int res = 0;
	for (auto a : nums) res ^= a;
	return res;
}

/* 137. Single Number II */
/* Given a non-empty array of integers, every element appears three times except for one, 
* which appears exactly once. Find that single one. */
int singleNumber(vector<int>& nums) {
	int res = 0; 
	for (int i = 0; i < 32; ++i) {
		int sum = 0;
		for (auto a : nums) {
			sum += (a >> i) & 1; 
		}
		res |= (sum % 3) << i;
	}
	return res; 
}

/* 393. UTF-8 Validation ??? */
/* A character in UTF8 can be from 1 to 4 bytes long, subjected to the following rules:
* For 1-byte character, the first bit is a 0, followed by its unicode code.
* For n-bytes character, the first n-bits are all one's, the n+1 bit is 0, 
* followed by n-1 bytes with most significant 2 bits being 10. Given an array of integers 
* representing the data, return whether it is a valid utf-8 encoding. Note: The input is an 
* array of integers. Only the least significant 8 bits of each integer is used to store the data. 
* This means each integer represents only 1 byte of data. Example: data = [197, 130, 1], 
* which represents the octet sequence: 11000101 10000010 00000001. Return true. */
bool validUtf8(vector<int>& data) {
	int n = data.size();
	for (int i = 0; i < n; ++i) {
		if (data[i] < 0b10000000) {
			continue;
		}
		else {
			int cnt = 0, val = data[i];
			for (int j = 7; j >= 1; --j) {
				if (val >= pow(2, j)) ++cnt;
				else break;
				val -= pow(2, j);
			}
			if (cnt == 1 || cnt > 4 || cnt > n - i) return false;
			for (int j = i + 1; j < i + cnt; ++j) {
				if (data[j] > 0b10111111 || data[j] < 0b10000000) return false;
			}
			i += cnt - 1;
		}
	}
	return true;
}


// ===========================================================

// ================ 12. Queue & Priority queue ===============
/* 373. Find K Pairs with Smallest Sums -- PRIORITY QUEUE */
/* You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.
* Define a pair (u,v) which consists of one element from the first array and one element
* from the second array. Find the k pairs (u1,v1),(u2,v2) ...(uk,vk) with the smallest sums. */
struct cmp {
	bool operator() (vector<int> &a, vector<int> &b) {
		return a[0] + a[1] < b[0] + b[1];
	}
};

vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
	vector<vector<int>> res;
	priority_queue<vector<int>, vector<vector<int>>, cmp> q;

	for (int i = 0; i < min((int)nums1.size(), k); ++i) {
		for (int j = 0; j < min((int)nums2.size(), k); ++j) {
			if (q.size() < k) {
				q.push({ nums1[i], nums2[j] });
			}
			else if (nums1[i] + nums2[j] < q.top()[0] + q.top()[1]) {
				q.push({ nums1[i], nums2[j] }); q.pop();
			}
		}
	}
	while (!q.empty()) {
		res.push_back(q.top()); q.pop();
	}
	return res;
}

/* 1167. Minimum Cost to Connect Sticks */
/* You have some sticks with positive integer lengths. You can connect any two sticks of lengths X and Y
* into one stick by paying a cost of X + Y.  You perform this action until there is one stick remaining.
* Return the minimum cost of connecting all the given sticks into one stick in this way.
* Input: sticks = [2,4,3]. Output: 14. */
int connectSticks(vector<int>& sticks) {
	int res = INT_MAX;
	priority_queue<int, vector<int>, greater<int>> q(sticks.begin(), sticks.end());
	while (q.size() > 1) {
		auto a = q.top(); q.pop();
		auto b = q.top(); q.pop();
		res += a + b;
		q.push(a + b);
	}
	return res;
}

/* 857. Minimum Cost to Hire K Workers */
/* There are N workers. The i-th worker has a quality[i] and a minimum wage expectation wage[i].
* Now we want to hire exactly K workers to form a paid group. When hiring a group of K workers,
* we must pay them according to the following rules:
* (1) Every worker in the paid group should be paid in the ratio of their quality compared to
*     other workers in the paid group.
* (2) Every worker in the paid group must be paid at least their minimum wage expectation.
* Return the least amount of money needed to form a paid group satisfying the above conditions.
* Input: quality = [10,20,5], wage = [70,50,30], K = 2. Output: 105.00000.
* Explanation: We pay 70 to 0-th worker and 35 to 2-th worker. */
double mincostToHireWorkers(vector<int>& quality, vector<int>& wage, int K) {
	int n = quality.size();
	double res = 10e9;
	vector<vector<double> > m; // use vector is better than "unordered_map" 
	for (int i = 0; i < n; ++i) {
		m.push_back({ (double)wage[i] / quality[i], (double)quality[i] });
	}
	sort(m.begin(), m.end());

	priority_queue<int> q;
	int sum = 0;
	for (auto a : m) {
		sum += a[1];
		q.push(a[1]);
		if (q.size() > K) {
			sum -= q.top();
			q.pop();
		}
		if (q.size() == K) {
			res = min(res, sum * a[0]);
		}
	}
	return res;
}

/* 871. Minimum Number of Refueling Stops */
/* A car travels from a starting position to a destination which is target miles east of the 
* starting position. Along the way, there are gas stations. Each station[i] represents 
* a gas station that is station[i][0] miles east of the starting position, 
* and has station[i][1] liters of gas. The car starts with an infinite tank of gas, 
* which initially has startFuel liters of fuel in it.  It uses 1 liter of gas per 1 mile that it drives.
* When the car reaches a gas station, it may stop and refuel, transferring all the gas 
* from the station into the car. What is the least number of refueling stops the car must
* make in order to reach its destination?  If it cannot reach the destination, return -1. 
* Input: target = 100, startFuel = 10, stations = [[10,60],[20,30],[30,30],[60,40]]. Output: 2. */
int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations) {
	int res = 0, cur = startFuel, n = stations.size(), i = 0; 
	priority_queue<int> q; 
	while (cur < target) {
		++res; 
		// Find all reachable stations using 'cur' gas, and save the gas into 'q'. 
		while (i < n && stations[i][0] <= cur) {
			q.push(stations[i++][1]);
		}
		if (q.empty()) return -1;
		// Just cumulate gas as moving forward. So 'cur' is the gas that can reach to current station.
		cur += q.top(); q.pop(); 
	}
	return res; 
}

/* 1199. Minimum Time to Build Blocks */
/* You are given a list of blocks, where blocks[i] = t means that the i-th block needs t units of time 
* to be built. A block can only be built by exactly one worker. A worker can either split into two workers 
* (number of workers increases by one) or build a block then go home. Both decisions cost some time.
* The time cost of spliting one worker into two workers is given as an integer split. 
* Note that if two workers split at the same time, they split in parallel so the cost would be split.
* Output the minimum time needed to build all blocks. Initially, there is only one worker. 
* Input: blocks = [1,2], split = 5. Output: 7. Input: blocks = [1,2,3], split = 1. Output: 4. */
int minBuildTime(vector<int>& blocks, int split) {
	priority_queue<int, vector<int>, greater<int>> q(blocks.begin(), blocks.end());
	while (q.size() > 1) {
		auto a = q.top(); q.pop(); 
		auto b = q.top(); q.pop(); 
		q.push(b + split);
	}
	return q.top(); 
}

/* 358. Rearrange String k Distance Apart */
/* Given a non-empty string s and an integer k, rearrange the string such that the 
* same characters are at least distance k from each other. All input strings are given
* in lowercase letters. If it is not possible to rearrange the string, return an empty string "".
* Example 1: Input: s = "aabbcc", k = 3. Output: "abcabc". */
string rearrangeString(string s, int k) {
	if (k == 0) return s; 
	string res("");
	unordered_map<char, int> m; 
	priority_queue<pair<int, char>> q; 
	for (auto c : s) ++m[c];
	for (auto it : m) q.push({ it.second, it.first });

	int n = s.size(); 
	while (!q.empty()) {
		int len = min(n, k);
		vector<pair<int, char>> vec; 

		for (int i = 0; i < len; ++i) {
			if (q.empty()) return "";
			auto t = q.top(); q.pop(); 
			res += t.second; 
			if (--t.first > 0) vec.push_back(t);
			--n;
		}
		for (auto a : vec) q.push(a);
	}
	return res; 
}

/* 506. Relative Ranks */
/* Given scores of N athletes, find their RELATIVE RANKS and the people with the top three highest scores,
* who will be awarded medals: "Gold Medal", "Silver Medal" and "Bronze Medal". 
* Example: [1, 2, 7, 5, 6, 3, 4], output: ["7","6","Gold Medal","Bronze Medal","Silver Medal","5","4"]. */
vector<string> findRelativeRanks(vector<int>& nums) {
	priority_queue<pair<int, int>> q; 
	int n = nums.size(), cnt = 1; 
	for (int i = 0; i < n; ++i) q.push({ nums[i], i }); // store number and index pair

	vector<string> res(n);
	for (int i = 0; i < n; ++i) {
		int ix = q.top().second; q.pop();

		if (cnt == 1) res[ix] = "Gold Medal";
		else if (cnt == 2) res[ix] = "Silver Medal";
		else if (cnt == 3) res[ix] = "Bronze Medal";
		else res[ix] = to_string(cnt);

		++cnt;
	}
	return res; 
}

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

// ===========================================================

// ================ 13. HASH MAP & MAP =======================

/* 217. Contains Duplicate -- ARRAY */
/* Given an array of integers, find if the array contains any duplicates.
* Input: [1,2,3,1]. Output: true. */
bool containsDuplicate(vector<int>& nums) {
	if (nums.empty()) return false;
	sort(nums.begin(), nums.end());
	for (int i = 0; i < nums.size() - 1; ++i) {
		if (nums[i] == nums[i + 1]) return true;
	}
	return false;
}

/* 219. Contains Duplicate II -- HASH MAP */
/* Given an array of integers and an integer k, find out whether there are two distinct
* indices i and j in the array such that nums[i] = nums[j] and the absolute difference
* between i and j is at most k.*/
bool containsNearbyDuplicate(vector<int>& nums, int k) {
	unordered_map<int, int> m;
	for (int i = 0; i < nums.size(); ++i) {
		if (m.find(nums[i]) != m.end() && i - m[nums[i]] <= k) return true;
		m[nums[i]] = i;
	}
	return false;
}

/* 220. Contains Duplicate III */
/* Given an array of integers, find out whether there are two distinct indices i and j in
* the array such that the absolute difference between nums[i] and nums[j] is at most t and
* the absolute difference between i and j is at most k. */
bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
	map<long long, int> m;
	int i = 0;
	for (int j = 0; j < nums.size(); ++j) {
		if (j - i > k) m.erase(nums[i++]);
		auto a = m.lower_bound((long long)nums[j] - t);

		if (a != m.end() && abs(a->first - (long long)nums[j]) <= t) return true;
		m[nums[j]] = j;
	}
	return false;
}

/* 525. Contiguous Array -- HASH MAP*/
/* Given a binary array, find the maximum length of a contiguous subarray
* with equal number of 0 and 1. [0, 1, 1, 0, 1, 0] */
int findMaxLength(vector<int>& nums) {
	// map stores unique sum as index. 
	unordered_map<int, int> m{ { 0, -1 } };
	int res = 0, n = nums.size(), sum = 0;

	for (int i = 0; i < n; ++i) {
		sum += nums[i] == 1 ? 1 : -1;
		if (m.count(sum)) res = max(res, i - m[sum]);
		else m[sum] = i;
	}
	return res;
}

/* 894. All Possible Full Binary Trees -- RECURSION */
/* A full binary tree is a binary tree where each node has exactly 0 or 2 children.
* Return a list of all possible full binary trees with N nodes.
* Each element of the answer is the root node of one possible tree.
* Each node of each tree in the answer MUST have node.val = 0.
* You may return the final list of trees in any order.
* RECURSION: start from N = 1, 3, 5, 7, etc. and find a pattern. */
vector<TreeNode*> allPossibleFBT(int N, unordered_map<int, vector<TreeNode*>>& m, vector<TreeNode*>& res) {
	if (m[N].size()) return m[N];
	if (N == 1) {
		res.push_back(new TreeNode(0));
	}
	else {
		for (int i = 1; i <= N; i += 2) {
			int l = i, r = N - 1 - i;
			vector<TreeNode*> left = allPossibleFBT(l);
			vector<TreeNode*> right = allPossibleFBT(r);

			for (auto a : left) {
				for (auto b : right) {
					TreeNode* root = new TreeNode(0);
					root->left = a;
					root->right = b;
					res.push_back(root);
				}
			}
		}
	}
	return m[N] = res;
}

vector<TreeNode*> allPossibleFBT(int N) {
	vector<TreeNode*> res;
	unordered_map<int, vector<TreeNode*> > m;
	return allPossibleFBT(N, m, res);
}

/* 327. Count of Range Sum */
/* Given an integer array nums, return the number of range sums that lie in [lower, upper] 
* inclusive. Range sum S(i, j) is defined as the sum of the elements in nums between 
* indices i and j (i ≤ j), inclusive. 
* Input: nums = [-2,5,-1], lower = -2, upper = 2, Output: 3. 
* 只有那些满足 lower <= sum[i] - sum[j] <= upper 的j能形成一个区间 [j, i] 满足题意. 
* lower_bound 是找数组中第一个不小于给定值的数的位置(包括等于情况)，
* upper_bound 是找数组中第一个大于给定值的数的位置 。 */
int countRangeSum(vector<int>& nums, int lower, int upper) {
	multiset<long long> st;
	st.insert(0);
	long long sum = 0;
	int res = 0;

	for (int i = 0; i < nums.size(); ++i) {
		sum += nums[i];
		res += distance(st.lower_bound(sum - upper), st.upper_bound(sum - lower));
		st.insert(sum);
	}
	return res;
}

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
			res = max(res, heights[t] * ( st.empty() ? i : (i - st.top() - 1)));
		}
		st.push(i);
	}
	return res; 
}

/* 150. Evaluate Reverse Polish Notation */
/* Evaluate the value of an arithmetic expression in Reverse Polish Notation.
* Valid operators are +, -, *, /. Each operand may be an integer or another expression. 
* Input: ["2", "1", "+", "3", "*"]. Output: 9. Explanation: ((2 + 1) * 3) = 9. */
int evalRPN(vector<string>& tokens) {
	stack<int> st; 	
	for (auto s : tokens) {
		if (s != "+" && s != "-" && s != "*" && s != "/") {
			st.push(stoi(s)); 
		}
		else {
			auto b = st.top(); st.pop(); 
			auto a = st.top(); st.pop(); 
			if (s == "+") st.push(a + b); 
			if (s == "-") st.push(a - b); 
			if (s == "*") st.push(a * b); 
			if (s == "/") st.push(a / b); 
		}
	}
	return st.top();
}

/* 636. Exclusive Time of Functions */
/* On a single threaded CPU, we execute some functions. Each function has a unique id 
* between 0 and N-1. We store logs in timestamp order that describe when a function 
* is entered or exited. Each log is a string with this format: 
* "{function_id}:{"start" | "end"}:{timestamp}".  For example, "0:start:3" means 
* the function with id 0 started at the beginning of timestamp 3.  "1:end:2" means 
* the function with id 1 ended at the end of timestamp 2. Return the exclusive time 
* of each function, sorted by their function id. 
* n = 2, logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]. Output: [3, 4]. */
vector<int> exclusiveTime(int n, vector<string>& logs) {
	vector<int> res(n, 0);
	stack<int> st; 
	int pre = 0; 

	for (auto s : logs) {
		int ix = s.find(":"), ix1 = s.find_last_of(":");
		int key = stoi(s.substr(0, ix)); 
		string type = s.substr(ix + 1, ix1 - ix - 1); 
		int time = stoi(s.substr(ix1 + 1));

		if (!st.empty()) {
			res[st.top()] += time - pre; 
		}

		pre = time; 
		if (type == "start") {
			st.push(key); 
		}
		else {
			auto t = st.top(); st.pop(); 
			++res[t]; 
			++pre; 
		}
	}
	return res; 
}

/* 833. Find And Replace in String */
/* Input: S = "abcd", indexes = [0,2], sources = ["a","cd"], targets = ["eee","ffff"]
* Output: "eeebffff"
* Input: S = "abcd", indexes = [0,2], sources = ["ab","ec"], targets = ["eee","ffff"]
* Output: "eeecd".
* Logic: should do backwards for replacement. */
string findReplaceString(string S, vector<int>& indexes, vector<string>& sources, vector<string>& targets) {
	vector<pair<int, int>> m;
	for (int i = 0; i < indexes.size(); ++i) {
		m.push_back({ indexes[i], i });
	}
	sort(m.rbegin(), m.rend());

	for (auto it : m) {
		int i = it.first, j = it.second;
		string source = sources[j], target = targets[j];
		if (S.substr(i, source.size()) == source) {
			S = S.substr(0, i) + target + S.substr(i + source.size());
		}
	}
	return S;
}

/* 609. Find Duplicate File in System */
/* Given a list of directory info including directory path, and all the files with contents in this directory, 
* you need to find out all the groups of duplicate files in the file system in terms of their paths. 
* The output is a list of group of duplicate file paths. For each group, it contains all the file paths of 
* the files that have the same content. A file path is a string that has the following format: 
* Input: ["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]
* Output:[["root/a/2.txt","root/c/d/4.txt","root/4.txt"],["root/a/1.txt","root/c/3.txt"]] */
vector<vector<string>> findDuplicate(vector<string>& paths) {
	unordered_map<string, vector<string>> m; 
	vector<vector<string>> res; 
	for (auto s : paths) {
		istringstream is(s);
		string t(""), pre("");
		is >> pre;

		while (is >> t) {
			int ix = t.find_last_of('(');
			string dir = pre + "/" + t.substr(0, ix);
			string content = t.substr(ix + 1, t.size() - ix - 2); 
			m[content].push_back(dir);
		}
	}
	for (auto it : m) {
		if(it.second.size() > 1) res.push_back(it.second);
	}
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

/* 387. First Unique Character in a String */
int firstUniqChar(string s) {
	unordered_map<char, int> m;
	for (auto c : s) ++m[c];

	for (int i = 0; i < s.size(); ++i) {
		if (m[s[i]] == 1) return i;
	}
	return -1;
}

/* 166. Fraction to Recurring Decimal 
 * Input: numerator = 2, denominator = 3. Output: "0.(6)". */
string fractionToDecimal(int numerator, int denominator) {
	if (numerator == 0 || (denominator == 0 && numerator == INT_MIN)) return "0";
	long up = long(abs(numerator)), down = long(abs(denominator));

	int s1 = numerator >= 0 ? 1 : -1; 
	int s2 = denominator >= 0 ? 1 : -1; 

	long ind = up / down, rem = up % down; 
	string res = to_string(ind);
	if (s1 * s2 == -1 && (ind > 0 || rem > 0)) res = "-" + res; 
	if (rem == 0) return res;
	
	res += ".";
	unordered_map<long, int> m;
	int pos = 0; 
	string t("");
	while (rem != 0) {
		if (m.find(rem) != m.end()) {
			t.insert(m[rem], "("); 
			t += ')';
			return res + t; 
		}
		m[rem] = pos; 
		t += to_string((rem * 10) / down);
		rem = (rem * 10) % 10;
		++pos;
	}
	return res + t; 
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

/* 249. Group Shifted Strings */
/* Given a string, we can "shift" each of its letter to its successive letter, 
* for example: "abc" -> "bcd". We can keep "shifting" which forms the sequence:
* "abc" -> "bcd" -> ... -> "xyz"
* Given a list of strings which contains only lowercase alphabets, group all 
* strings that belong to the same shifting sequence. 
* Input: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"],
* Output:[["abc","bcd","xyz"], ["az","ba"], ["acef"],["a","z"]]. */
vector<vector<string>> groupStrings(vector<string>& strings) {
	vector<vector<string>> res; 
	unordered_map<string, vector<string>> m;

	for (auto s : strings) {
		string t("");
		for (auto c : s) t += ((c - s[0]) + 26) % 26;
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

/* 835. Image Overlap */
/* Two images A and B are given, represented as binary, square matrices of the same size.
* (A binary matrix has only 0s and 1s as values.) We translate one image however we choose
* (sliding it left, right, up, or down any number of units), and place it on top of the other image.
* After, the overlap of this translation is the number of positions that have a 1 in both images.
* What is the largest possible overlap? */
int largestOverlap(vector<vector<int>>& A, vector<vector<int>>& B) {
	int res = 0; 
	vector<pair<int, int>> v1, v2; 
	for (int i = 0; i < A.size(); ++i) {
		for (int j = 0; j < A[i].size(); ++j) {
			if (A[i][j]) v1.push_back({ i, j });
		}
	}
	for (int i = 0; i < B.size(); ++i) {
		for (int j = 0; j < B[i].size(); ++j) {
			if (B[i][j]) v2.push_back({ i, j });
		}
	}; 
	unordered_map<string, int> m; 
	for (auto a : v1) {
		for (auto b : v2) {
			++m[to_string(a.first - b.first) + "_" + to_string(a.second - b.second)];
		}
	}
	for (auto it : m) {
		res = max(res, it.second);
	}
	return res; 
}

/* 582. Kill Process */
vector<int> killProcess(vector<int>& pid, vector<int>& ppid, int kill) {
	vector<int> res; 
	unordered_map<int, vector<int>> m; 
	int n = pid.size(); 
	for (int i = 0; i < n; ++i) {
		m[ppid[i]].push_back(pid[i]);
	}
	queue<int> q{ {kill} };
	while (!q.empty()) {
		auto t = q.front(); q.pop(); 
		res.push_back(t); 
		for (auto it : m[t]) {
			q.push(it);
		}
	}
	return res; 
}

/* 1090. Largest Values From Labels */
/* Input: values = [5,4,3,2,1], labels = [1,1,2,2,3], num_wanted = 3, use_limit = 1. Output: 9 */
int largestValsFromLabels(vector<int>& values, vector<int>& labels, int num_wanted, int use_limit) {
	int res = 0, n = values.size(); 
	multimap<int, int> m_map; 
	unordered_map<int, int> m;  

	for (int i = 0; i < n; ++i) m_map.insert({ values[i], labels[i] }); 
	for (auto it = m_map.rbegin(); it != m_map.rend() && num_wanted > 0; ++it) {
		if (++m[it->second] <= use_limit) {
			res += it->first; 
			--num_wanted; 
		}
	}
	return res; 
}

/* 388. Longest Absolute File Path -- HOT */
/* Suppose we abstract our file system by a string in the following manner:
* The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:
* dir
*    subdir1
*    subdir2
*        file.ext
* The directory dir contains an empty sub-directory subdir1 and a sub-directory subdir2 
* containing a file file.ext. */
int lengthLongestPath(string input) {
	unordered_map<int, int> m;  // {depth, length} mapping
	int n = input.size(), res = 0, depth = 0; 
	for (int i = 0; i < n; ++i) {
		int start = i; 
		while (i < n && input[i] != '\n' && input[i] != '\t') ++i;

		if (i == n || input[i] == '\n') {
			string s = input.substr(start, i - start);
			if (s.find(".") != string::npos) {
				res = max(res, m[depth] + (int)s.size());
			}
			else {
				++depth; 
				m[depth] = m[depth - 1] + (int)s.size() + 1; 
			}
			depth = 0; 
		}
		else if (input[i] == '\t') {
			++depth; 
		}
	}
	return res; 
}

/* 409. Longest Palindrome */
/* Given a string which consists of lowercase or uppercase letters, find the length of the 
* longest palindromes that can be built with those letters. This is case sensitive, 
* for example "Aa" is not considered a palindrome here. */
int longestPalindrome3(string s) {
	int res = 0, mid = 0; 
	unordered_map<char, int> m; 
	for (auto c : s) ++m[c];
	for (auto it : m) {
		res += it.second;
		if (it.second % 2 == 1) {
			res -= 1;
			mid = 1; 
		}
	}
	return mid == 1 ? 1 + res : res; 
}

/* 424. Longest Repeating Character Replacement */
/* Given a string s that consists of only uppercase English letters, 
* you can perform at most k operations on that string. Input:s = "ABAB", k = 2
* Output: 4. Input: s = "AABABBA", k = 1. Output: 4. */
int characterReplacement(string s, int k) {
	int res = 0, n = s.size(), start = 0, mxCnt = 0;
	int m[26] = { 0 };

	for (int i = 0; i < n; ++i) {
		mxCnt = max(mxCnt, ++m[s[i] - 'A']); // find max char
		while (i - start + 1 - mxCnt > k) { // maintain an rolling window 
			--m[s[start] - 'A'];
			++start;
		}
		res = max(res, i - start + 1);
	}
	return res;
}

/* 1048. Longest String Chain */
/* Given a list of words, each word consists of English lowercase letters.
* A word chain is a sequence of words [word_1, word_2, ..., word_k] with k >= 1, 
* where word_1 is a predecessor of word_2, word_2 is a predecessor of word_3, and so on. 
* Return the longest possible length of a word chain with words chosen from the given list of words.
* Input: ["a","b","ba","bca","bda","bdca"]. Output: 4.
* Explanation: one of the longest word chain is "a","ba","bda","bdca". */
int longestStrChain(vector<string>& words) {
	sort(words.begin(), words.end(), [](string a, string b) {
		return a.size() < b.size(); 
	});
	int res = 0;
	unordered_map<string, int> m; 
	for (auto s : words) {
		int cnt = 0; 
		for (int i = 0; i < s.size(); ++i) {
			string t = s.substr(0, i) + s.substr(i + 1);
			cnt = max(cnt, m[t] + 1);
		}
		m[s] = cnt;
	}
	for (auto it : m) {
		res = max(res, it.second);
	}
	return res; 
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

/* left 指向该无重复子串左边的起始位置的前一个，由于是前一个，所以初始化就是 -1，然后遍历整个字符串，
* 对于每一个遍历到的字符，如果该字符已经在 HashMap 中存在了，并且如果其映射值大于 left 的话，
* 那么更新 left 为当前映射值。然后映射值更新为当前坐标i，这样保证了 left 始终为当前边界的前一个位置 */
int lengthOfLongestSubstring2(string s) {
	int res = 0, left = -1, n = s.size();
	unordered_map<int, int> m;
	for (int i = 0; i < n; ++i) {
		if (m.count(s[i]) && m[s[i]] > left) {
			left = m[s[i]];
		}
		m[s[i]] = i;
		res = max(res, i - left);
	}
	return res;
}

/* 395. Longest Substring with At Least K Repeating Characters */
/* Find the length of the longest substring T of a given string (consists of lowercase letters only)
* such that every character in T appears no less than k times. Input: s = "aaabb", k = 3; Output:3 */
int longestSubstring(string s, int k) {
	int m[26] = { 0 }, i = 0, n = s.size();
	for (auto c : s) ++m[c - 'a'];
	while (i < n && m[s[i] - 'a'] >= k) ++i;
	if (i == n) return n;
	/* locate the first character that appear less than k times in the string.
	* this character is definitely not included in the result, and that
	* separates the string into two parts. */
	int left = longestSubstring(s.substr(0, i), k);
	int right = longestSubstring(s.substr(i + 1), k); // IMPORTANT. "i + 1"
	return max(left, right);
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

/* 32. Longest Valid Parentheses */
/* Given a string containing just the characters '(' and ')', find the length of the longest valid
* (well-formed) parentheses substring. Input: "(()". Output: 2 */
int longestValidParentheses(string s) {
	int res = 0, n = s.size(), start = 0;
	stack<int> st;
	for (int i = 0; i < n; ++i) {
		auto c = s[i];
		if (c == '(') st.push(i);
		else {
			if (st.empty()) start = i + 1;
			else {
				auto t = st.top(); st.pop();
				res = max(res, st.empty() ? i - start + 1 : i - st.top());
			}
		}
	}
	return res;
}

/* 720. Longest Word in Dictionary */
/* Given a list of strings words representing an English Dictionary, find the longest word in words
* that can be built one character at a time by other words in words. If there is more than one possible
* answer, return the longest word with the smallest lexicographical order.
* Input: words = ["w","wo","wor","worl", "world"]. Output: "world". Explanation:
The word "world" can be built one character at a time by "w", "wo", "wor", and "worl". */
string longestWord(vector<string>& words) {
	sort(words.begin(), words.end());
	unordered_set<string> st;
	string res("");

	for (auto a : words) {
		if (a.size() == 1 || st.count(a.substr(0, a.size() - 1))) {
			res = a.size() > res.size() ? a : res;
			st.insert(a);
		}
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

/* 325. Maximum Size Subarray Sum Equals k */
/* Given an array nums and a target value k, find the maximum length of a subarray that sums to k. 
* If there isn't one, return 0 instead. Input: nums = [1, -1, 5, -2, 3], k = 3. Output: 4  
* Input: nums = [-2, -1, 2, 1], k = 1. Output: 2. */
int maxSubArrayLen(vector<int>& nums, int k) {
	if (nums.empty()) return 0;
	unordered_map<int, int> m; // {sum, start position}
	int n = nums.size(), res = 0, sum = 0; 
	for (int i = 0; i < n; ++i) {
		sum += nums[i]; 
		if (sum == k) res = i + 1; 
		else {
			if (m.find(sum - k) != m.end()) res = max(res, i - m[sum - k]);
		}
		if (!m.count(sum)) m[sum] = i; 
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

/* 939. Minimum Area Rectangle */
/* Given a set of points in the xy-plane, determine the minimum area of a rectangle formed from these points,
 * with sides parallel to the x and y axes. If there isn't any rectangle, return 0. 
 * Example 1: Input: [[1,1],[1,3],[3,1],[3,3],[2,2]]. Output: 4. 
 * Example 2: Input: [[1,1],[1,3],[3,1],[3,3],[4,1],[4,3]]. Output: 2. */
int minAreaRect(vector<vector<int>>& points) {
	int res = INT_MAX; 
	unordered_map<int, set<int>> m; 
	for (auto a : points) {
		m[a[0]].insert(a[1]);
	}
	for (auto i = m.begin(); i != m.end(); ++i) {
		for (auto j = next(i); j != m.end(); ++j) {
			if (i->second.size() < 2 || j->second.size() < 2) continue; 
			vector<int> v; 
			set_intersection(begin(i->second), end(i->second), begin(j->second), end(j->second), back_inserter(v));
			for (int k = 1; k < v.size(); ++k) {
				res = min(res, abs(j->first - i->first) * (v[k] - v[k - 1]));
			}
		}
	}
	return res == INT_MAX ? 0 : res; 
}

/* 963. Minimum Area Rectangle II */
/* Given a set of points in the xy-plane, determine the minimum area of any rectangle formed from these points, 
* with sides not necessarily parallel to the x and y axes. If there isn't any rectangle, return 0 */
size_t d2(int x1, int y1, int x2, int y2) {
	return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

double minAreaFreeRect(vector<vector<int>>& points) {
	unordered_map<int, vector<vector<int>>> m;
	size_t res = 0;
	for (int i = 0; i < points.size(); ++i) {
		for (int j = i + 1; j < points.size(); ++j) {
			auto center = ((size_t)(points[i][0] + points[j][0]) << 16) + points[i][1] + points[j][1];
			m[center].push_back({ points[i][0], points[i][1], points[j][0], points[j][1] });
		}
	}
	for (auto it : m) {
		for (auto i = 0; i < it.second.size(); ++i) {
			for (auto j = i + 1; j < it.second.size(); ++j) {
				auto &p1 = it.second[i], &p2 = it.second[j];
				if ((p1[0] - p2[0]) * (p1[0] - p2[2]) + (p1[1] - p2[1]) * (p1[1] - p2[3]) == 0) {
					auto area = d2(p1[0], p1[1], p2[0], p2[1]) * d2(p1[0], p1[1], p2[2], p2[3]);
					if (res == 0 || res > area) res = area;
				}
			}
		}
	}
	return sqrt(res);
}

/* 726. Number of Atoms */
/* Given a formula, output the count of all elements as a string in the following form: the first name (in sorted order),
* followed by its count (if that count is more than 1), followed by the second name (in sorted order),
* followed by its count (if that count is more than 1), and so on. Example:  formula = "Mg(OH)2". Output: "H2MgO2". */
map<string, int> generate(string formula, int& i) {
	map<string, int> res;
	int n = formula.size();

	while (i < n) {
		// if is '(', use recursion and save recursion part into res
		if (formula[i] == '(') {
			++i;
			// recursion
			for (auto a : generate(formula, i)) {
				res[a.first] += a.second;
			}
		}
		// if is ')', find the number. 
		else if (formula[i] == ')') {
			int j = ++i;
			while (i < n && isdigit(formula[i])) ++i;
			int cnt = stoi(formula.substr(j, i - j));
			for (auto a : res) res[a.first] *= cnt;
			return res;
		}
		// else just save the element and cnt into map
		else {
			int j = i++;
			while (i < n && islower(formula[i])) ++i;
			string elem = formula.substr(j, i - j);
			j = i;
			while (i < n && isdigit(formula[i])) ++i;
			string cnt = formula.substr(j, i - j);
			res[elem] += cnt.empty() ? 1 : stoi(cnt);
		}
	}
	return res;
}

string countOfAtoms(string formula) {
	stack<int> st;
	int i = 0;
	string res("");
	map<string, int> m = generate(formula, i);
	for (auto it : m) {
		res += it.first + (it.second == 1 ? "" : to_string(it.second));
	}
	return res;
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

/* 465. Optimal Account Balancing */
/* A group of friends went on holiday and sometimes lent each other money. 
* For example, Alice paid for Bill's lunch for $10. Then later Chris gave Alice $5 for a taxi ride.
* We can model each transaction as a tuple (x, y, z) which means person x gave person y $z. 
* Assuming Alice, Bill, and Chris are person 0, 1, and 2 respectively (0, 1, 2 are the person's ID), 
* the transactions can be represented as [[0, 1, 10], [2, 0, 5]].
* Given a list of transactions between a group of people, return the minimum number of transactions 
* required to settle the debt. 
* Example: Input: [[0,1,10], [1,0,1], [1,2,5], [2,0,5]]. Output: 1. */
int minTransfers(vector<int>& accounts, int n, int ix, int num) {
	int res = INT_MAX; 
	while (ix < n && !accounts[ix]) ++ix; 
	
	for (int i = ix + 1; i < n; ++i) {
		if (accounts[i] * accounts[ix] < 0) {
			accounts[i] += accounts[ix];
			// Recursion.
			res = min(res, minTransfers(accounts, n, ix + 1, num + 1));
			accounts[i] -= accounts[ix];
		}
	}
	return res == INT_MAX ? num : res; 
}

int minTransfers(vector<vector<int>>& transactions) {
	unordered_map<int, int> m; 
	for (auto a : transactions) {
		m[a[0]] += a[2];
		m[a[1]] -= a[2];
	}
	int cnt = 0, n = m.size(); 
	vector<int> accounts(n, 0);
	for (auto it : m) {
		if (it.second) accounts[cnt++] = it.second; 
	}
	return minTransfers(accounts, cnt, 0, 0);
}

/* 336. Palindrome Pairs */
/* Given a list of unique words, find all pairs of distinct indices (i, j) in the given list, 
* so that the concatenation of the two words, i.e. words[i] + words[j] is a palindrome. 
* Example 1: Input: ["abcd","dcba","lls","s","sssll"], Output: [[0,1],[1,0],[3,2],[2,4]] 
* Explanation: The palindromes are ["dcbaabcd","abcddcba","slls","llssssll"]. */
bool isValidPal(string s, int left, int right) {
	if (left > right) return false; 
	while (left <= right) {
		if (s[left++] != s[right--]) return false; 
	}
	return true; 
}

vector<vector<int>> palindromePairs(vector<string>& words) {
	vector<vector<int>> res; 
	unordered_map<string, int> m; 
	set<int> st; 
	for (int i = 0; i < words.size(); ++i) {
		m[words[i]] = i; 
		st.insert(words[i].size()); 
	}
	for (int i = 0; i < words.size(); ++i) {
		string t = words[i];
		int len = t.size();
		reverse(t.begin(), t.end());
		if (m.count(t) && m[t] != i) {
			res.push_back({ i, m[t] });
		}
		auto a = st.find(len);
		for (auto it = st.begin(); it != a; ++it) {
			int d = *it;
			if (isValidPal(t, 0, len - d - 1) && m.count(t.substr(len - d))) {
				res.push_back({ i, m[t.substr(len - d)] });
			}
			if (isValidPal(t, d, len - 1) && m.count(t.substr(0, d))) {
				res.push_back({ m[t.substr(0, d)], i });
			}
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

// Solution2: Use two vectors to record the number of chars for s1, s2; 
bool checkInclusion(string s1, string s2) {
	int m = s1.size(), n = s2.size(); 
	if (m > n) return false; 
	vector<int> v1(128, 0), v2(128, 0);
	for (int i = 0; i < m; ++i) {
		++v1[s1[i]];
		++v2[s2[i]];
	}
	if (v1 == v2) return true;
	for (int i = m; i < n; ++i) {
		++v2[s2[i]];
		--v2[s2[i - m]];
		if (v1 == v2) return true;
	}
	return false;
}

/* 1122. Relative Sort Array */
/* Given two arrays arr1 and arr2, the elements of arr2 are distinct, and all elements in arr2
* are also in arr1. Sort the elements of arr1 such that the relative ordering of items in arr1
* are the same as in arr2.  Elements that don't appear in arr2 should be placed at the end of arr1
* in ascending order. Example: Input: arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
* Output: [2,2,2,1,4,3,3,9,6,7,19] */
vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
	map<int, int> m;
	vector<int> res;
	for (auto a : arr1) ++m[a];
	for (auto a : arr2) {
		for (int i = 0; i < m[a]; ++i) res.push_back(a);
		m.erase(a);
	}
	for (auto it : m) {
		for (int i = 0; i < it.second; ++i) res.push_back(it.first);
	}
	return res;
}

/* 1021. Remove Outermost Parentheses */
/* A valid parentheses string is either empty (""), "(" + A + ")", or A + B, where A and B are
* valid parentheses strings, and + represents string concatenation.  For example, "", "()", "(())()",
* and "(()(()))" are all valid parentheses strings. A valid parentheses string S is primitive if
* it is nonempty, and there does not exist a way to split it into S = A+B, with A and B nonempty
* valid parentheses strings. Given a valid parentheses string S, consider its primitive decomposition:
* S = P_1 + P_2 + ... + P_k, where P_i are primitive valid parentheses strings. Return S after removing
* the outermost parentheses of every primitive string in the primitive decomposition of S.
* Input: "(()())(())". Output: "()()()".
* LOGIC: opened count the number of opened parenthesis. Add every char to the result,
* unless the first left parenthesis and the last right parenthesis.*/
string removeOuterParentheses(string S) {
	int opened = 0;
	string res("");
	for (auto c : S) {
		if (c == '(' && opened++ > 0) res += c;
		if (c == ')' && opened-- > 1) res += c;
	}
	return res;
}

string removeOuterParentheses2(string S) {
	int n = S.size(), j = 0, cnt = 0;
	string res("");
	for (int i = 0; i < n; ++i) {
		j = i, cnt = 0;
		string t("");
		while (i == j || (j < n && cnt != 0)) {
			if (S[j] == '(') {
				++cnt;
				t += S[j++];
			}
			else if (S[j] == ')') {
				--cnt;
				t += S[j++];
			}
		}
		if (cnt == 0) res += t.substr(1, t.size() - 2);
		--j;
		i = j;
	}
	return res;
}

/* 1233. Remove Sub-Folders from the Filesystem */
/* Given a list of folders, remove all sub-folders in those folders and return in any order
* the folders after removing. If a folder[i] is located within another folder[j], it is called
* a sub-folder of it. Input: folder = ["/a","/a/b","/c/d","/c/d/e","/c/f"]. Output: ["/a","/c/d","/c/f"]. */
vector<string> removeSubfolders(vector<string>& folder) {
	vector<string> res;
	sort(folder.begin(), folder.end());
	for (auto s : folder) {
		if (res.empty() || s.rfind(res.back() + "/") == string::npos) {
			res.push_back(s);
		}
	}
	return res;
}

/* 1182. Shortest Distance to Target Color */
/* You are given an array colors, in which there are three colors: 1, 2 and 3.
* You are also given some queries. Each query consists of two integers i and c,
* return the shortest distance between the given index i and the target color c.
* If there is no solution return -1. Input: colors = [1,1,2,1,3,2,2,3,3],
* queries = [[1,3],[2,2],[6,1]]. Output: [3,0,3]. */
int shortestDistanceColorBS(vector<int>& nums, int target) {
	// HERE "nums" and "target" both mean index.
	int n = nums.size();
	int pos = lower_bound(nums.begin(), nums.end(), target) - nums.begin();
	if (pos == 0) return nums[0];
	if (pos == n) return nums[n - 1];

	if (nums[pos] - target < target - nums[pos - 1])
		return nums[pos];
	return nums[pos - 1];
}

vector<int> shortestDistanceColor(vector<int>& colors, vector<vector<int>>& queries) {
	vector<int> res;
	unordered_map<int, vector<int>> m;
	for (int i = 0; i < colors.size(); ++i) {
		m[colors[i]].push_back(i);
	}

	for (auto a : queries) {
		auto target = a[0];

		if (!m.count(a[1])) {
			res.push_back(-1);
			continue; // IMPORTANT.
		}
		int pos = shortestDistanceColorBS(m[a[1]], target);
		res.push_back(abs(pos - target));
	}
	return res;
}

/* 632. Smallest Range Covering Elements from K Lists */
/* You have k lists of sorted integers in ascending order. Find the smallest range that includes 
* at least one number from each of the k lists. We define the range [a,b] is smaller than range [c,d] 
* if b-a < d-c or a < c if b-a == d-c.
* Input: [[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]. Output: [20,24]. */
vector<int> smallestRange(vector<vector<int>>& nums) {
	vector<int> res;
	vector<pair<int, int>> v; // save all number and group index into one vector
	for (int i = 0; i < nums.size(); ++i) {
		for (auto a : nums[i]) {
			v.push_back({ a, i });
		}
	}
	sort(v.begin(), v.end());
	// "cnt" is the numbers of groups in the window.
	int left = 0, n = v.size(), k = nums.size(), diff = INT_MAX, cnt = 0;  
	unordered_map<int, int > m; // number and count mapping
	for (int right = 0; right < n; ++right) {
		if (m[v[right].second] == 0) ++cnt;
		++m[v[right].second];
		while (left <= right && cnt == k) {
			if (diff > v[right].first - v[left].first) {
				diff = v[right].first - v[left].first;
				res = { v[left].first, v[right].first };
			}
			if (--m[v[left].second] == 0) --cnt;
			++left; 
		}
	}
	return res; 
}

/* 659. Split Array into Consecutive Subsequences */
/* Given an array nums sorted in ascending order, return true if and only if you can split
* it into 1 or more subsequences such that each subsequence consists of consecutive integers 
* and has length at least 3. Input: [1,2,3,3,4,5]. Output: True. */
bool isPossible(vector<int>& nums) {
	unordered_map<int, int> freq, need;

	for (auto a : nums) ++freq[a];
	for (auto a : nums) {
		if (freq[a] == 0) continue;
		if (need[a] > 0) {
			--need[a];
			++need[a + 1];
		}
		else if (freq[a + 1] > 0 && freq[a + 2] > 0) {
			--freq[a + 1];
			--freq[a + 2];
			++need[a + 3];
		}
		else return false;
		--freq[a];
	}
	return true;
}

/* 1296. Divide Array in Sets of K Consecutive Numbers */
/* Given an array of integers nums and a positive integer k, find whether it's possible to divide 
* this array into sets of k consecutive numbers. Return True if its possible otherwise return False.
* Example 1: Input: nums = [1,2,3,3,4,4,5,6], k = 4. Output: true
* Explanation: Array can be divided into [1,2,3,4] and [3,4,5,6]. */
bool isPossibleDivide(vector<int>& nums, int k) {
	int n = nums.size();
	if (n % k != 0) return false;
	map<int, int> m;
	for (auto a : nums) ++m[a];
	for (auto it : m) {
		if (it.second) {
			int cnt = it.second;
			for (int i = 0; i < k; ++i) {
				if (m[it.first + i] < cnt) return false;
				else m[it.first + i] -= cnt;
			}
		}
	}
	return true;
}

/* 1153. String Transforms Into Another String */
/* Given two strings str1 and str2 of the same length, determine whether you can transform str1 into str2 
* by doing zero or more conversions. In one conversion you can convert all occurrences of one character 
* in str1 to any other lowercase English character. Return true if and only if you can transform str1 into str2.
* Example 1: str1 = "aabcc", str2 = "ccdee". Output: true. 
* Example 2: str1 = "abcdefghijklmnopqrstuvwxyz", str2 = "bcdefghijklmnopqrstuvwxyza" => false. */
bool canConvert(string str1, string str2) {
	if (str1 == str2) return true;
	unordered_map<char, char> m;
	set<char> st(str2.begin(), str2.end());

	for (int i = 0; i < str1.size(); ++i) {
		if (m.count(str1[i]) && m[str1[i]] != str2[i]) return false;
		m[str1[i]] = str2[i];
	}
	// In both case, there should at least one character that is unused, so total unique chars should be < 26.
	return st.size() < 26;
}

/* 246. Strobogrammatic Number */
/* A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
*  Write a function to determine if a number is strobogrammatic. The number is represented as a string. 
* Input:  "88". Output: true. */
bool isStrobogrammatic(string num) {
	unordered_map<char, char> m{ { '0', '0' },{ '1', '1' },{ '8', '8' },{ '6', '9' },{ '9', '6' } };
	int n = num.size(), i = 0, j = n - 1;
	while (i <= j) {
		if (!m.count(num[i]) || m[num[i]] != num[j]) return false;
		++i, --j;
	}
	return true;
}

/* 247. Strobogrammatic Number II */
/* A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
* Find all strobogrammatic numbers that are of length = n. Example: Input:  n = 2. Output: ["11","69","88","96"] */
vector<string> findStrobogrammatic(int m, int n) {
	vector<string> res;
	if (m == 0) return { "" };
	if (m == 1) return { "0", "1", "8" };
	vector<string> t = findStrobogrammatic(m - 2, n);
	for (auto a : t) {
		if (m != n) res.push_back("0" + a + "0");
		res.push_back("1" + a + "1");
		res.push_back("8" + a + "8");
		res.push_back("6" + a + "9");
		res.push_back("9" + a + "6");
	}
	return res; 
}

vector<string> findStrobogrammatic(int n) {
	return findStrobogrammatic(n, n);
}

/* 248. Strobogrammatic Number III */
/* A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
* Write a function to count the total strobogrammatic numbers that exist in the range of low <= num <= high.
* Example: Input: low = "50", high = "100". Output: 3. Explanation: 69, 88, and 96 are three strobogrammatic numbers. */
void strobogrammaticInRange(string low, string high, string path, int len, int& res) {
	if (path.size() >= len) {
		if (path.size() != len && (len != 1 && path[0] == '0')) return; 
		if ((len == low.size() && path.compare(low) < 0 ) || (len == high.size() && path.compare(high) > 0)) return; 
		++res;
	}
	strobogrammaticInRange(low, high, "0" + path + "0", len, res);
	strobogrammaticInRange(low, high, "1" + path + "1", len, res);
	strobogrammaticInRange(low, high, "6" + path + "9", len, res);
	strobogrammaticInRange(low, high, "8" + path + "8", len, res);
	strobogrammaticInRange(low, high, "9" + path + "6", len, res);
}

int strobogrammaticInRange(string low, string high) {
	int res = 0; 
	for (int i = low.size(); i <= high.size(); ++i) {
		strobogrammaticInRange(low, high, "0", i, res);
		strobogrammaticInRange(low, high, "1", i, res);
		strobogrammaticInRange(low, high, "8", i, res); 
	}
	return res; 
}

/* 218. The Skyline Problem -- hard  ??? */
/* A city's skyline is the outer contour of the silhouette formed by all the buildings in that city 
* when viewed from a distance. Now suppose you are given the locations and height of all the buildings 
* as shown on a cityscape photo (Figure A), write a program to output the skyline formed by these 
* buildings collectively. The geometric information of each building is represented by a triplet 
* of integers [Li, Ri, Hi], where Li and Ri are the x coordinates of the left and right edge of the 
* ith building, respectively, and Hi is its height. For instance, the dimensions of all buildings 
* in Figure A are recorded as: [ [2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8] ]. For instance, 
the skyline in Figure B should be represented as:[ [2 10], [3 15], [7 12], [12 0], [15 10], [20 8], [24, 0] ].*/
vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
	vector<vector<int> > res;
	vector<pair<int, int> > v;
	for (auto a : buildings) {
		v.push_back({ a[0], -a[2] });
		v.push_back({ a[1], a[2] });
	}

	sort(v.begin(), v.end());
	multiset<int> m;
	m.insert(0);
	int pre = 0, cur = 0;

	for (auto a : v) {
		if (a.second < 0) m.insert(-a.second);
		else m.erase(m.find(a.second));

		cur = *m.rbegin();
		if (pre != cur) {
			res.push_back({ a.first, cur });
			pre = cur;
		}
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

// ===========================================================

// ================ 14. TWO POINTERS   =======================

/* 1056. Confusing Number -- TWO POINTERS */
/* Given a number N, return true if and only if it is a confusing number,
* which satisfies the following condition: We can rotate digits by
* 180 degrees to form new digits. When 0, 1, 6, 8, 9 are rotated 180 degrees,
* they become 0, 1, 9, 8, 6 respectively. When 2, 3, 4, 5 and 7 are rotated 180 degrees,
* they become invalid. A confusing number is a number that when rotated 180 degrees
* becomes a different number with each digit valid. */
bool confusingNumber(int N) {
	unordered_map<char, char> m{ { '0', '0' },{ '1', '1' },{ '8', '8' },{ '6', '9' },{ '9', '6' } };

	string s = to_string(N), t = s;
	if (s.size() == 1) return N == 6 || N == 9;
	int i = 0, j = s.size() - 1;

	while (i <= j) {
		if (!m.count(t[i]) || !m.count(t[j])) return false;
		t[i] = m[t[i]], t[j] = m[t[j]];
		swap(t[i++], t[j--]);
	}
	return t != s;
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

/* 713. Subarray Product Less Than K */
/* Your are given an array of positive integers nums. Count and print the number of (contiguous)
* subarrays where the product of all the elements in the subarray is less than k.
* Example 1: Input: nums = [10, 5, 2, 6], k = 100. Output: 8 */
int numSubarrayProductLessThanK(vector<int>& nums, int k) {
	if (k <= 1) return 0;
	int left = 0, n = nums.size(), p = 1, res = 0;
	for (int i = 0; i < n; ++i) {
		p *= nums[i];
		// Maintain a window that has the the production equals to k
		// shrink the window to the right if "p" 
		while (p >= k) p /= nums[left++];
		res += i - left + 1;
	}
	return res;
}

// ===========================================================

// ================ 15. STRING PROBLEMS =======================

/* 405. Convert a Number to Hexadecimal -- BASE QUESTION */
/* All letters in hexadecimal (a-f) must be in lowercase.
* The hexadecimal string must not contain extra leading 0s. If the number is zero, it is 
* represented by a single zero character '0'; otherwise, the first character in the 
* hexadecimal string will not be the zero character. */
string toHex(int num) {
	if (num == 0) return "0";
	string res("");
	unsigned int t = num;
	unordered_map<int, string> m{ { 10, "a" },{ 11, "b" },{ 12, "c" },
	{ 13, "d" },{ 14, "e" },{ 15, "f" } };

	while (t > 0) {
		res = ((t % 16) >= 10 ? m[t % 16] : to_string(t % 16)) + res;
		t /= 16;
	}
	return res;
}

/* 504. Base 7 -- STRING & MATH */
/* Given an integer, return its base 7 string representation.
* Input: 100. Output: "202" .  Input: -7. Output: "-10". */
string convertToBase7(int num) {
	if (num == 0) return "0";
	string res("");
	long long t = abs(num);
	while (t > 0) {
		res = to_string(t % 7) + res;
		t /= 7;
	}
	return num > 0 ? res : "-" + res;
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

/* 1170. Compare Strings by Frequency of the Smallest Character -- STRING */
/* Let's define a function f(s) over a non-empty string s, which calculates the frequency of 
* the smallest character in s. For example, if s = "dcce" then f(s) = 2 because the smallest 
* character is "c" and its frequency is 2. Now, given string arrays queries and words, 
* return an integer array answer, where each answer[i] is the number of words such that 
* f(queries[i]) < f(W), where W is a word in words.
* Input: queries = ["bbb","cc"], words = ["a","aa","aaa","aaaa"]. Output: [1,2] */
int calFreq(string s) {
	map<char, int> m;
	for (auto c : s) ++m[c];
	return m.begin()->second;
}

vector<int> numSmallerByFrequency(vector<string>& queries, vector<string>& words) {
	int n = queries.size(), m = words.size();
	vector<int> v1(n, 0), v2(m, 0), res(n, 0);
	for (int i = 0; i < n; ++i) {
		v1[i] = calFreq(queries[i]);
	}
	for (int i = 0; i < m; ++i) {
		v2[i] = calFreq(words[i]);
	}
	sort(v2.begin(), v2.end());
	for (int i = 0; i < n; ++i) {
		res[i] = m - (upper_bound(v2.begin(), v2.end(), v1[i]) - v2.begin());
	}
	return res;
}

/* 38. Count and Say */
/* Given an integer n where 1 ≤ n ≤ 30, generate the nth term of the count-and-say sequence.
*  n = 5 =>  111221. */
string countAndSay(int n) {
	string res("1");
	while (--n) {
		string t (""); 
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

/* 165. Compare Version Numbers */
/* Compare two version numbers version1 and version2. If version1 > version2 return 1; 
* if version1 < version2 return -1;otherwise return 0. 
* Input: version1 = "7.5.2.4", version2 = "7.5.3". Output: -1. */
int compareVersion(string version1, string version2) {
	int m = version1.size(), n = version2.size(), i = 0, j = 0;
	string p, q; 
	while (i < m || j < n) {
		while (i < m && version1[i] != '.') p += version1[i++];
		while (j < n && version2[j] != '.') q += version2[j++];
	}
	int a = stoi(p), b = stoi(q); 
	if (a > b) return 1; 
	else if (a < b) return -1; 
	else {
		p.clear(); 
		q.clear(); 
		++i, ++j; 
	}
	return 0; 
}

/* 944. Delete Columns to Make Sorted */
/* Suppose we chose a set of deletion indices D such that after deletions, each remaining 
* column in A is in non-decreasing sorted order. */
int minDeletionSize(vector<string>& A) {
	int res = 0, m = A.size(), n = A[0].size();

	for (int j = 0; j < n; ++j) {
		int i = 0;
		while (i < m) {
			if (i < m - 1 && A[i][j] > A[i + 1][j]) {
				++res;
				break;
			}
			else {
				++i;
			}
		}
	}
	return res;
}

/* 67. Add Binary -- STRING */
/* Given two binary strings, return their sum (also a binary string).
* The input strings are both non-empty and contains only characters 1 or 0.
* Input: a = "11", b = "1". Output: "100" */
string addBinary(string a, string b) {
	int m = a.size(), n = b.size(), i = m - 1, j = n - 1, carry = 0;
	string res("");
	while (i >= 0 || j >= 0) {
		int p = i >= 0 ? a[i--] - '0' : 0;
		int q = j >= 0 ? b[j--] - '0' : 0;

		int sum = p + q + carry;
		carry = sum / 2;
		res = to_string(sum % 2) + res;

	}
	return carry == 1 ? "1" + res : res;
}

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

/* 616. Add Bold Tag in String & 758. Bold Words in String-- STRING */
/* Input: s = "abcxyz123". dict = ["abc","123"]. Output:"<b>abc</b>xyz<b>123</b>"*/
string addBoldTag(string s, vector<string>& dict) {
	int n = s.size();
	string res("");
	vector<int> v(n, 0);
	for (auto a : dict) {
		int len = a.size();
		for (int i = 0; i <= n - len; ++i) {
			if (s.substr(i, len) == a) {
				for (int j = i; j < i + len; ++j) {
					v[j] = 1;
				}
			}
		}
	}
	for (int i = 0; i < n; ++i) {
		if (v[i]) {
			if (i == 0 || !v[i - 1]) res += "<b>";
			res += s[i];
			if (i == n - 1 || !v[i + 1]) res += "</b>";
		}
		else {
			res += s[i];
		}
	}
	return res;
}

/* 1023. Camelcase Matching */
/* A query word matches a given pattern if we can insert lowercase letters to 
* the pattern word so that it equals the query.
* Input: queries = ["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"], 
* pattern = "FB". Output: [true,false,true,true,false] */
bool camelMatchHelper(string s, string p) {
	int n = p.size(), i = 0;
	for (auto c : s) {
		if (i < n && c == p[i]) ++i;
		else if (c < 'a') return false;
	}
	return i == n;
}

vector<bool> camelMatch(vector<string>& queries, string pattern) {
	vector<bool> res(queries.size(), 0);
	for (int i = 0; i < queries.size(); ++i) {
		res[i] = camelMatchHelper(queries[i], pattern);
	}
	return res;
}

/* 299. Bulls and Cows -- STRING && MAP */
/* Input: secret = "1807", guess = "7810". Output: "1A3B" */
string getHint(string secret, string guess) {
	if (secret.size() != guess.size()) return "";
	int n = secret.size(), bull = 0, cow = 0;
	unordered_map<char, int> m;

	for (int i = 0; i < n; ++i) {
		if (secret[i] == guess[i]) ++bull;
		else ++m[secret[i]];
	}
	for (int i = 0; i < n; ++i) {
		if (secret[i] != guess[i] && m[guess[i]]) {
			++cow;
			--m[guess[i]];
		}
	}
	return to_string(bull) + "A" + to_string(cow) + "B";
}

string getHint2(string secret, string guess) {
	int bull = 0, cow = 0, n = secret.size();
	int m[256] = { 0 };

	for (int i = 0; i < n; ++i) {
		if (secret[i] == guess[i]) ++bull;
		else {
			if (++m[guess[i]] <= 0) ++cow;
			if (--m[secret[i]] >= 0) ++cow;
		}
	}
	return to_string(bull) + "A" + to_string(cow) + "B";
}

/* 1138. Alphabet Board Path-- STRING */
/* On an alphabet board, we start at position (0, 0), corresponding to character 
* board[0][0]. Here, board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"].
* '!' adds the character board[r][c] at our current position (r, c) to the answer.
* Input: target = "leet". Output: "DDR!UURRR!!DDD!".  */
string alphabetBoardPath(string target) {
	string res("");
	int x0 = 0, y0 = 0; 

	for (auto c : target) {
		int x = (c - 'a') % 5, y = (c - 'a') / 5; 

		res += string(max(0, y0 - y), 'U') + string(max(0, x - x0), 'R') +
			string(max(0, x0 - x), 'L') + string(max(0, y - y0), 'D');
		res += '!';

		x0 = x, y0 = y;
	}
	return res; 
}

/* 844. Backspace String Compare -- STRING */
/* Given two strings S and T, return if they are equal when both are typed into empty 
* text editors. # means a backspace character. Input: S = "ab#c", T = "ad#c"
Output: true. Explanation: Both S and T become "ac".*/
string backspaceHelper(string s) {
	string res("");
	int n = s.size(), cnt = 0; 
	for (int i = n - 1; i >= 0; --i) {
		if (s[i] == '#') {
			++cnt;
		}
		else {
			if (cnt == 0) res += s[i]; 
			else --cnt;
		}
	}
	return res; 
}

bool backspaceCompare(string S, string T) {
	return backspaceHelper(S) == backspaceHelper(T);
}

/* 859. Buddy Strings */
/* Given two strings A and B of lowercase letters, return true if and only if we 
* can swap two letters in A so that the result equals B. */
bool buddyStrings(string A, string B) {
	if (A.size() != B.size()) return false;
	set<char> st(A.begin(), A.end()); // IMPORTANT.
	vector<int> diff;
	if (A == B) return st.size() < A.size();
	else {
		for (int i = 0; i < A.size(); ++i) {
			if (A[i] != B[i]) diff.push_back(i);
		}
		return diff.size() == 2 && (A[diff[0]] == B[diff[1]] && A[diff[1]] == B[diff[0]]);
	}
}

/* 942. DI String Match */
/* Given a string S that only contains "I" (increase) or "D" (decrease), let N = S.length.
* Return any permutation A of [0, 1, ..., N] such that for all i = 0, ..., N-1
* Input: "IDID". Output: [0,4,1,3,2]. Input: "III". Output: [0,1,2,3]. */
vector<int> diStringMatch(string S) {
	int n = S.size(); 
	vector<int> res(n + 1, 0);
	for (int i = 0; i <= n; ++i) res[i] = i;
	for (int i = 0; i < n; ++i) {
		if (S[i] == 'D') {
			int j = i;
			while (j < n && S[j] == 'D') ++j;
			reverse(res.begin() + i, res.begin() + j + 1);
			i = j;
		}
	}
	return res; 
}

/* 816. Ambiguous Coordinates -- STRING & RECURSION */
/* We had some 2-dimensional coordinates, like "(1, 3)" or "(2, 0.5)".  Then, we removed all 
* commas, decimal points, and spaces, and ended up with the string S.  Return a list of 
* strings representing all possibilities for what our original coordinates could have been.
* Input: "(123)". Output: ["(1, 23)", "(12, 3)", "(1.2, 3)", "(1, 2.3)"]
* Input: "(00011)". Output:  ["(0.001, 1)", "(0, 0.011)"]. 
* Input: "(0123)". Output: ["(0, 123)", "(0, 12.3)", "(0, 1.23)", "(0.1, 23)", "(0.1, 2.3)", "(0.12, 3)"]
* Input: "(100)". Output: [(10, 0)]*/
vector<string> ambiguousHelper(string s) {
	vector<string> res{ {s} };
	int n = s.size();

	if (n == 0 || (n > 1 && s[0] == '0' && s[1] == '0')) return {}; 
	if (n == 1 || s[0] == '0') return res; 

	for (int i = 1; i < n; ++i) {
		res.push_back(s.substr(0, i) + "." + s.substr(i));
	}
	return res; 
}

vector<string> ambiguousCoordinates(string S) {
	vector<string> res; 
	int n = S.size(); 
	for (int i = 1; i < n - 2; ++i) {
		vector<string> left = ambiguousHelper(S.substr(1, i));
		vector<string> right = ambiguousHelper(S.substr(i + 1, n - 2 - i));

		for (auto a : left) {
			for (auto b : right) {
				res.push_back("(" + a + ", " + b + ")");
			}
		}
	}
	return res; 
}

/* 1016. Binary String With Substrings Representing 1 To N -- STRING & BITSET */
/* Given a binary string S (a string consisting only of '0' and '1's) 
* and a positive integer N, return true if and only if for every integer 
* X from 1 to N, the binary representation of X is a SUBSTRING of S.
* Input: S = "0110", N = 3. Output: true. 
* Note: 1 <= S.length <= 1000. 1 <= N <= 10^9. */
bool queryString(string S, int N) {
	while (N > 0) {
		auto t = bitset<32>(N--).to_string();
		if (S.find(t.substr(t.find("1"))) == string::npos) return false; 
	}
	return true; 
}

/* 224. Basic Calculator -- HARD -- STRING */
/* 所有计算器问题都可以用下面这个模板，如果有括号，用递归算出括号的部分。 */
/* Implement a basic calculator to evaluate a simple expression string.
* The expression string may contain open ( and closing parentheses ),
* the plus + or minus sign -, non-negative integers and empty spaces .
* Input: "1 + 1". Output: 2. Input: "(1+(4+5+2)-3)+(6+8)". Output: 23. */
int calculate11(string s) {
	int n = s.size(), res = 0, cur = 0;
	long long num = 0;
	char sign = '+';

	for (int i = 0; i < n; ++i) {
		char c = s[i];
		if (c >= '0' && c <= '9') {
			num = num * 10 + c - '0';
		}
		else if (c == '(') {
			int j = i, cnt = 0;
			for (; i < n; ++i) {
				if (s[i] == '(') ++cnt;
				if (s[i] == ')') --cnt;
				if (cnt == 0) break;
			}
			num = calculate11(s.substr(j + 1, i - j - 1));
		}

		if (c == '+' || c == '-' || i == n - 1) {
			switch (sign) {
			case '+': cur += num; break;
			case '-': cur -= num; break;
			}
			res += cur;
			cur = 0;
			num = 0;
			sign = c;
		}
	}
	return res;
}

int calculate1(string s) {
	int res = 0, n = s.size(), sign = 1;
	stack<int> st;
	for (int i = 0; i < n; ++i) {
		if (s[i] >= '0' && s[i] <= '9') {
			int num = 0;
			while (i < n && s[i] >= '0' && s[i] <= '9') {
				num = num * 10 + (s[i++] - '0');
			}
			res += num * sign;
			--i;
		}
		else if (s[i] == '+') {
			sign = 1;
		}
		else if (s[i] == '-') {
			sign = -1;
		}
		else if (s[i] == '(') {
			st.push(res);
			st.push(sign);
			res = 0;
			sign = 1;
		}
		else if (s[i] == ')') {
			res *= st.top(); st.pop();
			res += st.top(); st.pop();
		}
	}
	return res;
}

/* 227. Basic Calculator II -- STACK */
/* Implement a basic calculator to evaluate a simple expression string.
* The expression string contains only non-negative integers, +, -, *, /
* operators and empty spaces . The integer division should truncate toward zero.
* Input: " 3+5 / 2 ". Output: 5. */
int calculate22(string s) {
	int n = s.size(), res = 0, cur = 0;
	long long num = 0;
	char sign = '+';

	for (int i = 0; i < n; ++i) {
		char c = s[i];
		if (c >= '0' && c <= '9') {
			num = num * 10 + c - '0';
		}
		if (c == '+' || c == '-' || c == '*' || c == '/' || i == n - 1) {
			switch (sign) {
			case '+': cur += num; break;
			case '-': cur -= num; break;
			case '*': cur *= num; break;
			case '/': cur /= num; break;
			}
			if (c == '+' || c == '-' || i == n - 1) {
				res += cur;
				cur = 0;
			}
			num = 0;
			sign = c;
		}
	}
	return res;
}

int calculate2(string s) {
	int n = s.size(), res = 0, num = 0;
	stack<int> st;
	char sign = '+';

	for (int i = 0; i < n; ++i) {
		if (s[i] >= '0' && s[i] <= '9') {
			num = 0;
			while (i < n && s[i] >= '0' && s[i] <= '9') {
				num = num * 10 + s[i++] - '0';
			}
		}
		if ((s[i] < '0' && s[i] != ' ') || i == n - 1) {
			if (sign == '+') st.push(num);
			if (sign == '-') st.push(-num);
			if (sign == '*' || sign == '/') {
				int t = sign == '*' ? st.top() * num : st.top() / num;
				st.pop();
				st.push(t);
			}
			sign = s[i];
			num = 0;
		}
	}
	while (!st.empty()) {
		res += st.top(); st.pop();
	}
	return res;
}

/* 772. Basic Calculator III -- HARD -- STACK */
/* The expression string may contain open ( and closing parentheses ),
* the plus + or minus sign -, non-negative integers and empty spaces .
* " 6-4 / 2 " = 4, "2*(5+5*2)/3+(6/2+8)" = 21, "(2+6* 3+5- (3*14/7+2)*5)+3"=-12. */
int calculate3(string s) {
	int n = s.size(), res = 0, cur = 0;
	long long num = 0;
	char sign = '+';

	for (int i = 0; i < n; ++i) {
		char c = s[i];
		if (c >= '0' && c <= '9') {
			num = num * 10 + (c - '0');
		}
		else if (c == '(') {
			int j = i, cnt = 0;
			for (; i < n; ++i) {
				if (s[i] == '(') ++cnt;
				if (s[i] == ')') --cnt;
				if (cnt == 0) break;
			} // SCOPE!!!
			num = calculate3(s.substr(j + 1, i - j - 1));
		}

		if (c == '+' || c == '-' || c == '*' || c == '/' || i == n - 1) {
			switch (sign) {
			case '+': cur += num; break;
			case '-': cur -= num; break;
			case '*': cur *= num; break;
			case '/': cur /= num; break;
			}

			if (c == '+' || c == '-' || i == n - 1) {
				res += cur;
				cur = 0;
			}
			sign = c;
			num = 0;
		}
	}
	return res;
}

/* 770. Basic Calculator IV -- HARD */
/*   */

/* 241. Different Ways to Add Parentheses - DIVIDE & CONQUER */
/* Given a string of numbers and operators, return all possible results from computing all the 
* different possible ways to group numbers and operators. The valid operators are +, - and * .
* Input: "2*3-4*5". Output: [-34, -14, -10, -10, 10]. */
vector<int> diffWaysToCompute(string input) {
	vector<int> res; 
	int n = input.size(); 
	for (int i = 0; i < n; ++i) {
		if (input[i] == '+' || input[i] == '-' || input[i] == '*') {
			vector<int> left = diffWaysToCompute(input.substr(0, i));
			vector<int> right = diffWaysToCompute(input.substr(i + 1)); 

			for (auto a : left) {
				for (auto b : right) {
					if (input[i] == '+') res.push_back(a + b);
					else if (input[i] == '-') res.push_back(a - b);
					else if (input[i] == '*') res.push_back(a * b);
				}
			}
		}
	}
	if (res.empty()) res.push_back(stoi(input.c_str()));
	return res.size() ? res : vector<int>{ stoi(input) };
}

/* 282. Expression Add Operators -- HARD */
/* Given a string that contains only digits 0-9 and a target value, 
* return all possibilities to add binary operators (not unary) +, -, or * 
* between the digits so they evaluate to the target value. 
* Input: num = "123", target = 6. Output: ["1+2+3", "1*2*3"]. */
void addOperators(string num, int target, long diff, long cursum, string ind, vector<string>& res) {
	if (num.size() == 0 && cursum == target) {
		res.push_back(ind); 
	}

	for (int i = 1; i <= num.size(); ++i) {
		string cur = num.substr(0, i); 
		if (cur.size() > 0 && cur[0] == '0') return; 
		string next = num.substr(i); 

		if (ind.size() > 0) {
			addOperators(next, target, stoll(cur), cursum + stoll(cur), ind + "+" + cur, res);
			addOperators(next, target, -stoll(cur), cursum - stoll(cur), ind + "-" + cur, res);
			addOperators(next, target, diff * stoll(cur), (cursum - diff) + diff * stoll(cur), ind + "*" + cur, res);
		}
		else {
			addOperators(next, target, stoll(cur), stoll(cur), cur, res);
		}
	}
}

vector<string> addOperators(string num, int target) {
	vector<string> res; 
	addOperators(num, target, 0, 0, "", res);
	return res; 
}

/* 809. Expressive Words */
/* Given a list of query words, return the number of words that are stretchy. 
 * For some given string S, a query word is stretchy if it can be made to be 
 * equal to S by any number of applications of the following extension operation: 
 * choose a group consisting of characters c, and add some number of characters c 
 * to the group so that the size of the group is 3 or more.
 * Input: S = "heeellooo", words = ["hello", "hi", "helo"]. Output: 1.
 * TIPS: EXCELLENT CODE TO KEEP IN MIND.  */
bool expressiveWords(string s, string p) {
	if (p.size() > s.size()) return false;
	int j = 0, m = p.size(), n = s.size(); 

	for (int i = 0; i < n; i++) {
		if (j < m && s[i] == p[j]) j++;
		else if (i > 1 && s[i - 2] == s[i - 1] && s[i - 1] == s[i]);
		else if (0 < i && i < n - 1 && s[i - 1] == s[i] && s[i] == s[i + 1]);
		else return false;
	}
	return j == m;
}

int expressiveWords(string S, vector<string>& words) {
	int res = 0; 
	for (auto a : words) {
		if (expressiveWords(S, a)) ++res;
	}
	return res; 
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

/* 890. Find and Replace Pattern */
/* You have a list of words and a pattern, and you want to know which words in words matches the pattern.
* Input: words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb". Output: ["mee","aqq"]. */
bool isPattern(string s, string p) {
	if (s.size() != p.size()) return false; 
	int v1[128] = { -1 }, v2[128] = { -1 };
	for (int i = 0; i < s.size(); ++i) {
		if (v1[s[i]] != v2[p[i]]) return false; 
		v1[s[i]] = i + 1; 
		v2[p[i]] = i + 1;
	}
	return true; 
}

vector<string> findAndReplacePattern(vector<string>& words, string pattern) {
	vector<string> res; 
	for (auto s : words) {
		if (isPattern(s, pattern)) res.push_back(s);
	}
	return res; 
}

/* 564. Find the Closest Palindrome */
/* Given an integer n, find the closest integer (not including itself), which is a palindrome.
* The 'closest' is defined as absolute difference minimized between two integers.
* Input: "123". Output: "121". */
string nearestPalindromic(string n) {
	long len = n.size(), num = stoi(n), res, minDiff = LONG_MAX;
	unordered_set<long> st; 
	// If n = 3, the boundary of the qualified number is [99, 1001].
	st.insert(pow(10, len) + 1); 
	st.insert(pow(10, len - 1) - 1); 
	// To find the closest, find from the back. First need find the half.
	long prefix = stol(n.substr(0, (len + 1) / 2));

	// Two cases: 1. If given string is pal, eg. "121" -> "131"
	// 2. If given string is not pal: (1) odd length (2) even length. 
	for (int i = -1; i <= 1; ++i) {
		string pre = to_string(prefix + i);
		string s = pre + string(pre.rbegin() + (len & 1), pre.rend()); 
		st.insert(stol(s)); 
	}
	// Remove the original number first before update for minimum. 
	st.erase(num); 
	for (auto a : st) {
		long diff = abs(a - num);
		if (diff < minDiff) {
			minDiff = diff; 
			res = a; 
		}
		else if (diff == minDiff) {
			res = min(res, a);
		}
	}
	return to_string(res);
}

/* 293. Flip Game */
/* Input: s = "++++". Output: ["--++", "+--+", "++--"]. */
vector<string> generatePossibleNextMoves(string s) {
	vector<string> res;
	if (s.empty()) return res;
	int n = s.size();
	for (int i = 0; i < n - 1; ++i) {
		if (s[i] == '+' && s[i + 1] == '+') res.push_back(s.substr(0, i) + "--" + s.substr(i + 2));
	}
	return res;
}

/* 294. Flip Game II  */
/* Write a function to determine if the starting player can guarantee a win. */
bool canWin(string s) {
	int n = s.size(); 
	for (int i = 0; i < n - 1; ++i) {
		if (s[i] == '+' && s[i + 1] == '+' && !canWin(s.substr(0, i) + "--" + s.substr(i + 2)))
			return true; 
	}
	return false; 
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

/* 482. License Key Formatting */
/* Input: S = "5F3Z-2e-9-w", K = 4. Output: "5F3Z-2E9W" */
string licenseKeyFormatting(string S, int K) {
	string res("");
	int n = S.size(), cnt = 0;
	for (int i = n - 1; i >= 0; --i) {
		auto c = S[i];
		if (islower(c)) c -= 32;
		if (c == '-') continue;
		res += c;
		++cnt;
		if (cnt % K == 0)  res += '-';
	}
	if (res.back() == '-') res.pop_back();
	reverse(res.begin(), res.end());
	return res;
}

/* 481. Magical String */
/* A magical string S consists of only '1' and '2' and obeys the following rules:
 *The string S is magical because concatenating the number of contiguous occurrences of 
 * characters '1' and '2' generates the string S itself. Given an integer N as input, 
 * return the number of '1's in the first N number in the magical string S. 
 * The first few elements of string S is the following: S = "1221121221221121122……" */
int magicalString(int n) {
	string s("122");
	int i = 2; 
	while (i < n) {
		s.append(s[i] - '0', s.back() ^ 3); 
		++i; 
	}
	return count(s.begin(), s.begin() + n, '1');
}

/* 819. Most Common Word */
/* Given a paragraph and a list of banned words, return the most frequent word that is 
* not in the list of banned words. It is guaranteed there is at least one word that isn't banned, 
* and that the answer is unique. Words in the list of banned words are given in lowercase, 
* and free of punctuation.  Words in the paragraph are not case sensitive. The answer is in lowercase. 
* Input: paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
* banned = ["hit"]， Output: "ball". */
string mostCommonWord(string paragraph, vector<string>& banned) {
	unordered_set<string> dict(banned.begin(), banned.end());
	unordered_map<string, int> m;

	for (auto &c : paragraph) c = isalpha(c) ? tolower(c) : ' ';
	istringstream iss(paragraph);
	string s;
	int mx = 0;
	string res("");

	while (iss >> s) {
		if (!dict.count(s)) {
			mx = max(mx, ++m[s]);
		}
	}
	for (auto it : m) {
		if (it.second == mx) res = it.first;
	}
	return res;
}

string mostCommonWord2(string paragraph, vector<string>& banned) {
	unordered_set<string> dict(banned.begin(), banned.end());
	unordered_map<string, int> m;
	// IMPORATNT: pass by reference here
	for (auto &c : paragraph) c = isalpha(c) ? tolower(c) : ' ';
	istringstream iss(paragraph);
	string s;

	pair<string, int> res("", 0);

	while (iss >> s) {
		if (dict.find(s) == dict.end() && ++m[s] > res.second) {
			res = { s, m[s] };
		}
	}
	return res.first;
}

/* 43. Multiply Strings */
/* Given two non-negative integers num1 and num2 represented as strings, 
* return the product of num1 and num2, also represented as a string.
* Input: num1 = "123", num2 = "456". Output: "56088"*/
string multiply(string num1, string num2) {
	string res = "";
	int m = num1.size(), n = num2.size();
	vector<int> vals(m + n);

	for (int i = m - 1; i >= 0; --i) {
		for (int j = n - 1; j >= 0; --j) {
			int mul = (num1[i] - '0') * (num2[j] - '0');
			int p1 = i + j, p2 = i + j + 1, sum = mul + vals[p2];
			vals[p1] += sum / 10;
			vals[p2] = sum % 10;
		}
	}

	for (int val : vals) {
		if (!res.empty() || val != 0) res.push_back(val + '0');
	}
	return res.empty() ? "0" : res;
}

/* 681. Next Closest Time */
/* Given a time represented in the format "HH:MM", form the next closest time by reusing the 
* current digits. There is no limit on how many times a digit can be reused. You may assume the 
* given input string is always valid. For example, "01:34", "12:09" are all valid. 
* "1:34", "12:9" are all invalid. Example 1: Input: "19:34", Output: "19:39". */
string nextClosestTime(string time) {
	string res = time;
	set<int> st{ time[0], time[1], time[3], time[4] };
	string s(st.begin(), st.end());
	int n = s.size(); // IMPORTANT: difference bwtn 'res' and 's'. 

for (int i = res.size() - 1; i >= 0; --i) {
	if (res[i] == ':') continue;
	auto pos = s.find(res[i]);

	if (pos == n - 1) {
		res[i] = s[0];
	}
	else {
		char next = s[pos + 1];

		if ((i == 4) ||
			(i == 3 && next <= '5') ||
			(i == 1 && (res[0] != '2' || (res[0] == '2' && next <= '3'))) ||
			(i == 0 && next <= '2')) {
			res[i] = next;
			return res;
		}
		res[i] = s[0];
	}
}
return res;
}

/* 806. Number of Lines To Write String */
/* We are to write the letters of a given string S, from left to right into lines.
* Each line has maximum width 100 units, and if writing a letter would cause the width
* of the line to exceed 100 units, it is written on the next line.
* We are given an array widths, an array where widths[0] is the width of 'a',
* widths[1] is the width of 'b', ..., and widths[25] is the width of 'z'.
* Now answer two questions: how many lines have at least one character from S,
* and what is the width used by the last such line? Return your answer as
* an integer list of length 2.
* Example:Input: widths = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
* S = "abcdefghijklmnopqrstuvwxyz". Output: [3, 60].
* Input: widths = [4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
* S = "bbbcccdddaaa", Output: [2, 4]. */
vector<int> numberOfLines(vector<int>& widths, string S) {
	int res = 1, cur = 0; // 'res' is how many lines need, 'cur' current length
	for (auto c : S) {
		int len = widths[c - 'a'];
		if (cur + len > 100) ++res;
		cur = (cur + len > 100) ? len : cur + len;
	}
	return { res, cur };
}

/* 544. Output Contest Matches */
/* The n teams are given in the form of positive integers from 1 to n,
* which represents their initial rank. (Rank 1 is the strongest team and Rank n is the weakest team.)
* We'll use parentheses('(', ')') and commas(',') to represent the contest team pairing -
* parentheses('(' , ')') for pairing and commas(',') for partition.
* During the pairing process in each round, you always need to follow the strategy of making
* the rather strong one pair with the rather weak one. Example: Input: 4. Output: ((1,4),(2,3)). */
string findContestMatch(int n) {
	vector<string> v(n);
	for (int i = 0; i < n; ++i) v[i] = to_string(i + 1);

	while (n > 1) {
		for (int i = 0; i < n / 2; ++i) {
			v[i] = '(' + v[i] + ',' + v[n - 1 - i] + ')';
		}
		n /= 2;
	}
	return v[0];
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

/* 830. Positions of Large Groups */
/* In a string S of lowercase letters, these letters form consecutive groups of the same character.
* For example, a string like S = "abbxxxxzyy" has the groups "a", "bb", "xxxx", "z" and "yy".
* Call a group large if it has 3 or more characters.  We would like the starting and ending positions
* of every large group. */
vector<vector<int>> largeGroupPositions(string S) {
	vector<vector<int>> res;
	int n = S.size();
	for (int i = 0; i < n; ++i) {
		int j = i;
		while (i < n - 1 && S[i] == S[i + 1]) ++i;
		if (i - j >= 2) res.push_back({ j, i });
	}
	return res;
}

/* 838. Push Dominoes -- Two pointers */
/* There are N dominoes in a line, and we place each domino vertically upright.
 * Input: ".L.R...LR..L..". Output: "LL.RR.LLRRLL.."*/
string pushDominoes(string dominoes) {
	string s = 'L' + dominoes + 'R', res("");
	for (int i = 0, j = 1; j < s.size() ; ++j) {
		if (s[j] == '.') continue; 
		int mid = j - i - 1; 
		if (i > 0) res += s[i];

		if (s[i] == s[j]) res += string(mid, s[i]);
		else if (s[i] == 'L' && s[j] == 'R') {
			res += string(mid, '.');
		}
		else {
			res += string(mid / 2, 'R') + string(mid % 2, '.') + string(mid / 2, 'L');
		}
		i = j;
	}
	return res; 
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

/* 748. Shortest Completing Word */
/* Find the minimum length word from a given dictionary words, which has all the letters from the 
* string licensePlate. Such a word is said to complete the given string licensePlate.
* licensePlate will contain digits, spaces, or letters (uppercase or lowercase).
* words will have a length in the range [10, 1000].
* Every words[i] will consist of lowercase letters, and have length in range [1, 15].
* Eg: Input: licensePlate = "1s3 PSt", words = ["step", "steps", "stripe", "stepple"]. Output: "steps". */
string shortestCompletingWord(string licensePlate, vector<string>& words) {
	string res("");
	vector<int> v(26, 0);
	for (auto c : licensePlate) {
		if (isalpha(c)) {
			++v[tolower(c) - 'a'];
		}
	}
	int mn = INT_MAX;
	for (auto s : words) {
		vector<int> t(26, 0);
		for (auto c : s) {
			++t[c - 'a'];
		}
		bool b = true;
		for (int i = 0; i < t.size(); ++i) {
			if (t[i] < v[i]) {
				b = false;
				break;
			}
		}
		if (b && s.size() < mn) {
			res = s;
			mn = s.size();
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

/* 906. Super Palindromes */
/* Let's say a positive integer is a superpalindrome if it is a palindrome, and it is 
* also the square of a palindrome. Now, given two positive integers L and R (represented as strings), 
* return the number of superpalindromes in the inclusive range [L, R]. Example 1:
* Input: L = "4", R = "1000". Output: 4. */
bool isPalindrome2(string s) {
	int i = 0, j = s.size() - 1; 
	while (i <= j) {
		if (s[i++] != s[j--]) return false; 
	}
	return true;
}

void superpalindromesInRange(string cur, long left, long right, int& res) {
	if (cur.size() > 9) return; 
	if (!cur.empty() && cur[0] != '0') {
		long num = stol(cur);
		num *= num;
		if (num > right) return; 
		if (num >= left && isPalindrome2(to_string(num))) ++res; 
	}
	for (auto c = '0'; c <= '9'; ++c) {
		superpalindromesInRange(string(1, c) + cur + string(1, c), left, right, res);
	}
}
// Instead of brute force testing each number within the range.
// We can construct strings and test them with "abba" type and "abcba" type. 
int superpalindromesInRange(string L, string R) {
	int res = 0; 
	long left = stol(L), right = stol(R); 
	// "abba" type palindrome
	superpalindromesInRange("", left, right, res);
	// "abcba" type palindrome
	for (auto c = '0'; c <= '9'; ++c) {
		superpalindromesInRange(string(1, c), left, right, res);
	}
	return res; 
}

/* 68. Text Justification -- hard */
/* Given an array of words and a width maxWidth, format the text such that each line 
* has exactly maxWidth characters and is fully (left and right) justified. You should 
* pack your words in a greedy approach; that is, pack as many words as you can in each line. 
* Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.
* Extra spaces between words should be distributed as evenly as possible. 
* If the number of spaces on a line do not divide evenly between words, the empty slots 
* on the left will be assigned more spaces than the slots on the right.
* For the last line of text, it should be left justified and no extra space is inserted between words.*/
vector<string> fullJustify(vector<string>& words, int maxWidth) {
	vector<string> res;
	if (words.empty()) return res;

	int n = words.size(), i = 0;

	while (i < n) {
		int j = i, len = 0;
		while (j < n && len + words[j].size() + j - i <= maxWidth) {
			len += words[j++].size();
		}

		int space = maxWidth - len;
		string ind("");
		for (int k = i; k < j; ++k) {
			ind += words[k];
			int cnt = 0;

			if (space > 0) {
				if (j == n) {
					if (j - k == 1) cnt = space;
					else cnt = 1;
				}
				else {
					if (j - k > 1) {
						if (space % (j - k - 1) == 0) cnt = space / (j - k - 1);
						else cnt = space / (j - k - 1) + 1;
					}
					else {
						cnt = space;
					}
				}
			}

			ind.append(cnt, ' ');
			space -= cnt;
		}
		res.push_back(ind);
		i = j;
	}
	return res;
}

/* 929. Unique Email Addresses */
/* Every email consists of a local name and a domain name, separated by the @ sign.
* For example, in alice@leetcode.com, alice is the local name, and leetcode.com is the domain name.
* Besides lowercase letters, these emails may contain '.'s or '+'s. If you add periods ('.') 
* between some characters in the local name part of an email address, mail sent there will be forwarded 
* to the same address without dots in the local name. If you add a plus ('+') in the local name, 
* everything after the first plus sign will be ignored. 
* Input: ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
* Output: 2. Explanation: "testemail@leetcode.com" and "testemail@lee.tcode.com" actually receive mails. */
string remove(string s) {
	string res("");
	bool at = false, plus = false;
	for (auto c : s) {
		if (c == '@') {
			at = true;
		}
		else if (c == '+') {
			if (!at) {
				plus = true;
				continue;
			}
		}
		else if (c == '.') {
			if (!at) {
				continue;
			}
		}
		if (!at && plus) continue;
		res += c;
	}
	return res;
}

int numUniqueEmails(vector<string>& emails) {
	unordered_set<string> s;
	for (auto a : emails) {
		string t("");
		t = remove(a);
		s.insert(t);
	}
	return s.size();
}


// ===========================================================

// ================ 16. ARRAY PROBLEMS =======================
/* 66. Plus One */
/* Given a non-empty array of digits representing a non-negative integer, plus one to the integer.
* The digits are stored such that the most significant digit is at the head of the list, 
* and each element in the array contain a single digit. You may assume the integer does not contain 
* any leading zero, except the number 0 itself. Input: [1,2,3]. Output: [1,2,4]. */
vector<int> plusOne(vector<int>& digits) {
	int carry = 0, n = digits.size();
	for (int i = n - 1; i >= 0; --i) {
		if (digits[i] == 9) {
			digits[i] = 0; 
			carry = 1; 
		}
		else {
			++digits[i]; 
			return digits; 
		}
	}
	if (carry == 1) digits.insert(digits.begin(), 1); 
	return digits; 
}

/* 283. Move Zeroes */
/* Input: [0,1,0,3,12]. Output: [1,3,12,0,0] */
void moveZeroes(vector<int>& nums) {
	for (int i = 0, j = 0; i < nums.size(); ++i) {
		if (nums[i]) swap(nums[i], nums[j++]);
	}
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

/* 27. Remove Element */
/* Given an array nums and a value val, remove all instances of that value in-place and 
* return the new length. Do not allocate extra space for another array, you must do this 
* by modifying the input array in-place with O(1) extra memory. Given nums = [3,2,2,3], val = 3,
* Your function should return length = 2 */
int removeElement(vector<int>& nums, int val) {
	int res = 0, n = nums.size();
	for (int i = 0; i < n; ++i) {
		if (nums[i] != val) nums[res++] = nums[i];
	}
	return res;
}

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

/* 457. Circular Array Loop -- ARRAY ?? */
bool circularArrayLoop(vector<int>& nums) {
	unordered_map<int, int> m;
	int n = nums.size();
	vector<bool> visited(n, false);

	for (int i = 0; i < n; ++i) {
		if (visited[i]) continue;
		int cur = i;

		while (true) {
			visited[cur] = true;
			int next = (cur + nums[cur]) % n;
			if (next < 0) next += n;
			if (next == cur || nums[cur] * nums[next] < 0) break;

			if (m.count(next)) return true;
			m[cur] = next;
			cur = next;
		}
	}
	return false;
}

/* 523. Continuous Subarray Sum */
/* Given a list of non-negative numbers and a target integer k, write a function to 
* check if the array has a continuous subarray of size at least 2 that sums up to 
* a multiple of k, that is, sums up to n*k where n is also an integer. */
bool checkSubarraySum(vector<int>& nums, int k) {
	if (nums.size() < 2) return false;
	int sum = 0;

	for (int i = 0; i < nums.size(); ++i) {
		sum = nums[i];
		for (int j = i + 1; j < nums.size(); ++j) {
			sum += nums[j];
			if (sum == k) return true;
			if (k != 0 && sum % k == 0) return true;
		}
	}
	return false;
}

/* 565. Array Nesting -- ARRAY */
/* A zero-indexed array A of length N contains all integers from 0 to N-1.
* Find and return the longest length of set S, where S[i] = {A[i], A[A[i]], A[A[A[i]]], ... }
* subjected to the rule below. we stop adding right before a duplicate element occurs in S.
* Input: A = [5,4,0,3,1,6,2]. Output: 4. */
int arrayNesting(vector<int>& nums) {
	int res = 0, n = nums.size();
	vector<int> visited(n, 0);

	for (int i = 0; i < n; ++i) {
		if (visited[i]) continue;
		int pre = nums[i], cnt = 0;

		while (cnt == 0 || pre != nums[i]) {
			++cnt;
			visited[i] = 1;
			i = nums[i];
		}
		res = max(res, cnt);
	}
	return res;
}

/* 954. Array of Doubled Pairs -- ARRAY && HASH MAP */
/* Given an array of integers A with even length, return true if and only if
* it is possible to reorder it such that A[2 * i + 1] = 2 * A[2 * i]
* for every 0 <= i < len(A) / 2. Input: [3,1,3,6]. Output: false.
* Input: [4,-2,2,-4]. Output: true. */
bool canReorderDoubled(vector<int>& A) {
	int n = A.size();
	unordered_map<int, int> m;
	vector<int> keys;
	for (auto a : A) ++m[a];
	for (auto it : m) keys.push_back(it.first);
	// important part
	sort(keys.begin(), keys.end(), [](int a, int b) {
		return abs(a) < abs(b);
	});

	for (auto a : keys) {
		if (m[a] > m[2 * a]) return false;
		m[2 * a] -= m[a];
	}
	return true;
}

/* 932. Beautiful Array -- ARRAY ??? */
/* For some fixed N, an array A is beautiful if it is a permutation of the
* integers 1, 2, ..., N, such that: For EVERY i < j, there is NO k with
* i < k < j such that A[k] * 2 = A[i] + A[j]. Given N, return any beautiful array A.
* Input: 4. Output: [2,1,4,3]. Input: 5. Output: [3,1,2,5,4]. */
vector<int> beautifulArray(int N) {
	vector<int> res = { 1 };
	while (res.size() < N) {
		vector<int> t;
		for (auto a : res) if (2 * a - 1 <= N) t.push_back(2 * a - 1);
		for (auto a : res) if (2 * a <= N) t.push_back(2 * a);

		res = t;
	}
	return res;
}

/* 667. Beautiful Arrangement II -- ARRAY && TWO POINTERS */
/* Given two integers n and k, you need to construct a list which contains
* n different positive integers ranging from 1 to n and obeys the following
* requirement: Suppose this list is [a1, a2, a3, ... , an], then the list
* [|a1 - a2|, |a2 - a3|, |a3 - a4|, ... , |an-1 - an|] has exactly k
* distinct integers. If there are multiple answers, print any of them.
* Input: n = 3, k = 1. Output: [1, 2, 3]. Input: n = 3, k = 2. Output: [1, 3, 2].*/
vector<int> constructArray(int n, int k) {
	int i = 1, j = n;
	vector<int> res;
	while (i <= j) {
		while (k > 1) {
			res.push_back((k % 2 == 1) ? i++ : j--);
			--k;
		}
		res.push_back(i++);
	}
	return res;
}

/* 296. Best Meeting Point -- HARD -- MATH & SORT */
/* A group of two or more people wants to meet and minimize the total travel
* distance. You are given a 2D grid of values 0 or 1, where each 1 marks the home
* of someone in the group. The distance is calculated using Manhattan Distance,
* where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|. */
int manhattanDist(vector<int>& v) {
	sort(v.begin(), v.end());
	int res = 0, i = 0, j = v.size() - 1;
	while (i <= j) {
		res += (v[j--] - v[i++]);
	}
	return res;
}

int minTotalDistance(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size(), res = 0;
	vector<int> v1, v2;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == 1) {
				v1.push_back(i);
				v2.push_back(j);
			}
		}
	}
	return manhattanDist(v1) + manhattanDist(v2);
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

/* 469. Convex Polygon -- ARRAY & MATH */
/* Given a list of points that form a polygon when joined sequentially, find if this polygon is convex. */
bool isConvex(vector<vector<int>>& points) {
	// The cross product of two vectors and is a vector, with the property 
	// that it is orthogonal to the two vectors.
	long long n = points.size(), pre = 0, cur = 0; 
	for (int i = 0; i < n; ++i) {
		int dx1 = points[(i + 1) % n][0] - points[i % n][0];
		int dy1 = points[(i + 1) % n][1] - points[i % n][1];

		int dx2 = points[(i + 2) % n][0] - points[(i + 1) % n][0];
		int dy2 = points[(i + 2) % n][1] - points[(i + 1) % n][1];

		cur = dx1 * dy2 - dx2 * dy1;
		if (cur != 0) {
			if (pre * cur < 0) return false;
			pre = cur;
		}
	}
	return true; 
}

/* 498. Diagonal Traverse */
/* Given a matrix of M x N elements (M rows, N columns), return all elements of the matrix in diagonal 
* order as shown.Input: [[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ]]. Output:  [1,2,4,7,5,3,6,8,9].*/ 
vector<int> findDiagonalOrder(vector<vector<int>>& matrix) {
	if (matrix.empty() || matrix[0].empty()) return {};
	int m = matrix.size(), n = matrix[0].size(); 
	vector<int> res(m * n, 0);
	int dx[2] = { -1, 1 }, dy[2] = { 1, -1 }, dir = 0, r = 0, c = 0;

	for (int i = 0; i < res.size(); ++i) {
		res[i] = matrix[r][c];
		r += dx[dir], c += dy[dir];
		if (c >= n) { r += 2; c -= 1; dir = 1 - dir; }
		if (r >= m) { c += 2; r -= 1; dir = 1 - dir; }
		if (r < 0) { r = 0; dir = 1 - dir;}
		if (c < 0) { c = 0; dir = 1 - dir; }
	}
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

/* 321. Create Maximum Number -- HARD -- VECTOR */
/* Given two arrays of length m and n with digits 0-9
* representing two numbers. Create the MAXIMUM number of
* length k <= m + n from digits of the two. The relative order
* of the digits from the same array must be preserved. Return
* an array of the k digits.
* Note: You should try to optimize your time and space complexity.
* Input: nums1 = [3, 4, 6, 5], nums2 = [9, 1, 2, 5, 8, 3], k = 5
* Output: [9, 8, 6, 5, 3]. */
vector<int> mergeVector(vector<int> nums1, vector<int> nums2) {
	vector<int> res;
	while (!nums1.empty() || !nums2.empty()) {
		vector<int> &tmp = (nums1 > nums2) ? nums1 : nums2;
		res.push_back(tmp[0]);
		tmp.erase(tmp.begin());
	}
	return res;
}

vector<int> maxVector(vector<int>& nums, int k) {
	vector<int> res;
	int n = nums.size(), drop = n - k;
	for (auto a : nums) {
		while (drop > 0 && !res.empty() && a > res.back()) {
			res.pop_back();
			--drop;
		}
		res.push_back(a);
	}
	res.resize(k);
	return res;
}

vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
	int n1 = nums1.size(), n2 = nums2.size();
	vector<int> res;
	for (int i = max(0, k - n2); i <= min(k, n1); ++i) {
		res = max(res, mergeVector(maxVector(nums1, i), maxVector(nums2, k - i)));
	}
	return res;
}

/* 1144. Decrease Elements To Make Array Zigzag */
/* Given an array nums of integers, a move consists of choosing any element and decreasing it by 1.
 * Return the minimum number of moves to transform the given array nums into a zigzag array. 
 * Input: nums = [1,2,3]. Output: 2. Input: nums = [9,6,1,6,2]. Output: 4. 
 * Logic: Two options, either make A[even] smaller or make A[odd] smaller.
 * Loop on the whole array A, find the min(A[i - 1],A[i + 1]),
 * calculate that the moves need to make smaller than both side.
* If it's negative, it means it's already smaller than both side, no moved needed.
* Add the moves need to res[i % 2]. In the end return the smaller option. */
int movesToMakeZigzag(vector<int>& nums) {
	int res[2] = { 0, 0 }, n = nums.size(), left, right;
	for (int i = 0; i < n; ++i) {
		left = i > 0 ? nums[i - 1] : 1001;
		right = i + 1 < n ? nums[i + 1] : 1001;
		res[i % 2] += max(0, nums[i] - min(left, right) + 1);
	}
	return min(res[0], res[1]);
}

/* 1108. Defanging an IP Address */
/* A defanged IP address replaces every period "." with "[.]". */
string defangIPaddr(string address) {
	string res("");
	for (auto c : address) {
		if (c == '.') res += "[.]";
		else res += c;
	}
	return res;
}

/* 1184. Distance Between Bus Stops */
/* A bus has n stops numbered from 0 to n - 1 that form a circle. We know the distance 
* between all pairs of neighboring stops where distance[i] is the distance between the 
* stops number i and (i + 1) % n. The bus goes along both directions 
* i.e. clockwise and counterclockwise.
* Return the shortest distance between the given start and destination stops. */
int distanceBetweenBusStops(vector<int>& distance, int start, int destination) {
	if (start > destination) swap(start, destination); // IMPORTANT. 
	int total = accumulate(distance.begin(), distance.end(), 0);
	int t = accumulate(distance.begin() + start, distance.begin() + destination, 0); 
	return min(t, total - t); 
}

/* 1121. Divide Array Into Increasing Sequences -- HARD */
/* Given a non-decreasing array of positive integers nums and an integer K, 
* find out if this array can be divided into one or more disjoint increasing 
*subsequences of length at least K.
* Input: nums = [1,2,2,3,3,4,4], K = 3. Output: true.
* Logic: find the maximum quantity of same numbers in A. And this number should be 
* assigned to each group. so we can get the max group number. */
bool canDivideIntoSubsequences(vector<int>& nums, int K) {
	int n = nums.size(), cur = 0, group = 0;
	for (int i = 0; i < n - 1; ++i) {
		cur = nums[i] < nums[i + 1] ? 1 : cur + 1;
		group = max(group, cur);
	}
	return n >= K * group;
}

/* 1089. Duplicate Zeros */
/* Given a fixed length array arr of integers, duplicate each occurrence of zero, 
* shifting the remaining elements to the right. Note that elements beyond the 
* length of the original array are not written. Do the above modifications to 
* the input array in place, do not return anything from your function.
* Input: [1,0,2,3,0,4,5,0] modified to: [1,0,0,2,3,0,0,4]. 
* Logic: update from back to front. */
void duplicateZeros(vector<int>& arr) {
	int n = arr.size(), shift = 0, i = 0;
	for (i = 0; i + shift < n; ++i) shift += (arr[i] == 0);
	--i; 
	for (; shift > 0; --i) {
		if (i + shift < n) arr[i + shift] = arr[i]; 
		if (arr[i] == 0) arr[i + --shift] = arr[i];
	}
}

/* 789. Escape The Ghosts */
/* You are playing a simplified Pacman game. You start at the point (0, 0), 
* and your destination is (target[0], target[1]). There are several ghosts on the map, 
* the i-th ghost starts at (ghosts[i][0], ghosts[i][1]). Each turn, you and all ghosts 
* simultaneously *may* move in one of 4 cardinal directions: north, east, west, or south, 
* going from the previous point to a new point 1 unit of distance away.
* You escape if and only if you can reach the target before any ghost reaches you
* (for any given moves the ghosts may take.)  If you reach any square (including the target)
* at the same time as a ghost, it doesn't count as an escape.
* Return True if and only if it is possible to escape. */
bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target) {
	int dist = abs(target[0]) + abs(target[1]);
	int mn = INT_MAX; 

	for (auto a : ghosts) {
		int t = abs(a[0] - target[0]) + abs(a[1] - target[1]);
		mn = min(mn, t);
	}
	return dist < mn; 
}

/* 442. Find All Duplicates in an Array */
/* Given an array of integers, 1 ≤ a[i] ≤ n (n = size of array), 
* some elements appear twice and others appear once. Find all the elements 
* that appear twice in this array. Could you do it without extra space 
* and in O(n) runtime? */
vector<int> findDuplicates(vector<int>& nums) {
	vector<int> res;
	for (int i = 0; i < nums.size(); ++i) {
		int idx = abs(nums[i]) - 1;
		if (nums[idx] > 0) nums[idx] = -nums[idx];
		else res.push_back(idx + 1);
	}
	return res;
}

/* 448. Find All Numbers Disappeared in an Array */
/* Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array),
* some elements appear twice and others appear once. Find all the elements 
* of [1, n] inclusive that do not appear in this array. Could you do it without
* extra space and in O(n) runtime? You may assume the returned list does 
* not count as extra space. */
vector<int> findDisappearedNumbers(vector<int>& nums) {
	vector<int> res;
	for (int i = 0; i < nums.size(); ++i) {
		int idx = abs(nums[i]) - 1;
		if (nums[idx] > 0) nums[idx] = -nums[idx];
	}
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] > 0) res.push_back(i + 1);
	}
	return res;
}

/* 760. Find Anagram Mappings */
/* Given two lists Aand B, and B is an anagram of A. B is an anagram 
* of A means B is made by randomizing the order of the elements in A.
* We want to find an index mapping P, from A to B. A mapping P[i] = j 
* means the ith element in A appears in B at index j.
* These lists A and B may contain duplicates. If there are multiple 
* answers, output any of them. */
vector<int> anagramMappings(vector<int>& A, vector<int>& B) {
	if (A.size() != B.size()) return {};
	int n = A.size();
	vector<int> res(n, 0);
	unordered_map<int, int> m;

	for (int i = 0; i < n; ++i) {
		m[B[i]] = i;
	}

	for (int i = 0; i < n; ++i) {
		res[i] = m[A[i]];
	}
	return res;
}

/* 724. Find Pivot Index */
int pivotIndex(vector<int>& nums) {
	int n = nums.size();
	for (int i = 0; i < n; ++i) {
		int left = accumulate(nums.begin(), nums.begin() + i, 0);
		int right = accumulate(nums.begin() + i + 1, nums.end(), 0);
		if (left == right) return i;
	}
	return -1;
}

/* 389. Find the Difference */
char findTheDifference(string s, string t) {
	char res = 0;
	for (auto c : s) res ^= c;
	for (auto c : t) res ^= c;
	return res;
}

/*997.  Find the Town Judge */
/* In a town, there are N people labelled from 1 to N. There is a rumor that one 
* of these people is secretly the town judge. If the town judge exists, then:
* The town judge trusts nobody. Everybody (except for the town judge) trusts the town judge.
* There is exactly one person that satisfies properties 1 and 2.
* You are given trust, an array of pairs trust[i] = [a, b] representing that the person 
* labelled a trusts the person labelled b. */
int findJudge(int N, vector<vector<int>>& trust) {
	vector<int> truster(N + 1), trusted(N + 1);
	int n = trust.size();

	for (auto a : trust) {
		++truster[a[0]];
		++trusted[a[1]];
	}
	for (int i = 1; i <= N; ++i) {
		if (truster[i] == 0 && trusted[i] == N - 1) return i;
	}
	return -1;
}

/* 832. Flipping an Image */
/* Given a binary matrix A, we want to flip the image horizontally, then invert it, 
* and return the resulting image. Input: [[1,1,0],[1,0,1],[0,0,0]]. Output: [[1,0,0],[0,1,0],[1,1,1]]. */
vector<vector<int>> flipAndInvertImage(vector<vector<int>>& A) {
	int m = A.size(), n = A[0].size();
	vector<vector<int>> res(m, vector<int>(n, 0));
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			res[i][j] = A[i][n - 1 - j] ^ 1;
		}
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

/* 89. Gray Code */
/* The gray code is a binary numeral system where two successive values differ in only one bit. */
vector<int> grayCode(int n) {
	vector<int> res; 
	for (int i = 0; i < pow(2, n); ++i) {
		res.push_back((i >> 1) ^ i);
	}
	return res; 
}

/* 274. H-Index */
int hIndex(vector<int>& citations) {
	sort(citations.begin(), citations.end(), greater<int>());
	for (int i = 0; i < citations.size(); ++i) {
		if (i >= citations[i]) return i;
	}
	return citations.size();
}

/* 334. Increasing Triplet Subsequence */
/* Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.
* Formally the function should: Return true if there exists i, j, k, such that arr[i] < arr[j] < arr[k]
* given 0 ≤ i < j < k ≤ n-1 else return false. Note: Your algorithm should run in O(n) time complexity 
* and O(1) space complexity. Input: [1,2,3,4,5]. Output: true. */
bool increasingTriplet(vector<int>& nums) {
	int mn1 = INT_MAX, mn2 = INT_MAX;
	for (auto a : nums) {
		if (a <= mn1) {
			mn1 = a; 
		}
		else if (a > mn1 && a <= mn2) {
			mn2 = a; 
		}
		else {
			return true;
		}
	}
	return false;
}

/* 57. Insert Interval */
/* Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
* You may assume that the intervals were initially sorted according to their start times.
* Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]. Output: [[1,2],[3,10],[12,16]]*/
vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
	vector<vector<int>> res; 
	int n = intervals.size(), i = 0; 
	while (i < n && intervals[i][1] < newInterval[0]) {
		res.push_back(intervals[i++]);
	}
	while (i < n && intervals[i][0] <= newInterval[1]) {
		newInterval[0] = min(intervals[i][0], newInterval[0]);
		newInterval[1] = max(intervals[i][1], newInterval[1]);
		++i; 
	}
	res.push_back(newInterval); 
	while (i < n) {
		res.push_back(intervals[i++]);
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

/* 349. Intersection of Two Arrays */
/* Given two arrays, write a function to compute their intersection.
* Example 1: Input: nums1 = [1,2,2,1], nums2 = [2,2]. Output: [2]. */
vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
	set<int> st1(nums1.begin(), nums1.end()), st2;
	for (auto a : nums2) {
		st2.insert(a);
	}
	return vector<int>(st2.begin(), st2.end());
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

/* 986. Interval List Intersections */
/* Find all intersection intervals between two lists of intervals. */
vector<Interval> intervalIntersections(vector<Interval>& A, vector<Interval>& B) {
	vector<Interval> res; 
	int m = A.size(), n = B.size(), i = 0, j = 0; 
	while (i < m && j < n) {
		auto a = A[i], b = B[j]; 
		if (a.end < b.start) ++i; 
		else if (a.start > b.end) ++j; 
		else {
			res.push_back({ max(a.start, b.start), min(a.end, b.end) });
			if (a.end <= b.end) ++i;
			else ++j;
		}
	}
	return res;
}

/* 973. K Closest Points to Origin */
/* We have a list of points on the plane.  Find the K closest points to the origin (0, 0). */
vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
	sort(points.begin(), points.end(), [](vector<int>& a, vector<int>& b) {
		return a[0] * a[0] + a[1] * a[1] < b[0] * b[0] + b[1] * b[1];
	});
	return vector<vector<int>>(points.begin(), points.begin() + K);
}

/* 747. Largest Number At Least Twice of Others */
/* In a given integer array nums, there is always exactly one largest element.
* Find whether the largest element in the array is at least twice as much as
* every other number in the array. If it is, return the index of the largest element,
* otherwise return -1 */
int dominantIndex(vector<int>& nums) {
	if (nums.size() <= 1) return 0;
	int mx = INT_MIN, mx2 = INT_MIN, res = -1;
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] > mx) {
			mx2 = mx;
			mx = nums[i];
			res = i;
		}
		else if (nums[i] > mx2) {
			mx2 = nums[i];
		}
	}
	return mx >= mx2 + mx2 ? res : -1;
}

/* 562. Longest Line of Consecutive One in Matrix */
/* Given a 01 matrix M, find the longest line of consecutive one in the matrix. 
* The line could be horizontal, vertical, diagonal or anti-diagonal. */
int longestLine(vector<vector<int>>& M) {
	if (M.empty() || M[0].empty()) return 0;
	int res = 0, m = M.size(), n = M[0].size(); 
	vector<vector<int>> dirs{ {0, 1}, {1, 0}, {1, 1}, {-1, 1} };
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			for (auto dir : dirs) {
				int x = i, y = j, cnt = 0; 
				while (x >= 0 && x < m && y >= 0 && y < n && M[x][y] == 1) {
					++cnt;
					x += dir[0], y += dir[1];
				}
				res = max(res, cnt);
			}
		}
	}
	return res; 
}

/* 845. Longest Mountain in Array */
/* Given an array A of integers, return the length of the longest mountain. 
* Return 0 if there is no mountain. */
int longestMountain(vector<int>& A) {
	int res = 0, up = 0, down = 0; 
	for (int i = 1; i < A.size(); ++i) {
		if ((down && A[i] > A[i - 1]) || A[i - 1] == A[i]) {
			up = 0, down = 0; 
		}
		up += (A[i] > A[i - 1]);
		down += (A[i] < A[i - 1]);
		if (up && down) res = max(res, up + down + 1);
	}
	return res; 
}

/* 769. Max Chunks To Make Sorted */
/* Given an array arr that is a permutation of [0, 1, ..., arr.length - 1], we split the array into
* some number of "chunks" (partitions), and individually sort each chunk.  After concatenating them, 
* the result equals the sorted array. What is the most number of chunks we could have made? */
int maxChunksToSorted(vector<int>& arr) {
	int mx = INT_MIN, n = arr.size(), res = 0;
	for (int i = 0; i < n; ++i) {
		mx = max(mx, arr[i]);
		if (i == mx) ++res; 
	}
	return res; 
}

/* 768. Max Chunks To Make Sorted II */
/* Difference: this arr can  */
int maxChunksToSorted2(vector<int>& nums) {
	vector<int> t(nums);
	sort(t.begin(), t.end());
	int sum1 = 0, sum2 = 0, res = 0; 

	for (int i = 0; i < nums.size(); ++i) {
		sum1 += t[i]; 
		sum2 += nums[i];
		if (sum1 == sum2) ++res; 
	}
	return res; 
}

/* 485. Max Consecutive Ones */
/* Given a binary array, find the maximum number of consecutive 1s in this array.
* Input: [1,1,0,1,1,1]. Output: 3 */
int findMaxConsecutiveOnes(vector<int>& nums) {
	int res = 0, cnt = 0; 
	for (auto a : nums) {
		if (a == 1) {
			res = max(res, ++cnt); 
		}
		else {
			cnt = 0; 
		}
	}
	return res; 
}

/* 487. Max Consecutive Ones II */
/* Given a binary array, find the maximum number of consecutive 1s in this array if you can flip
* at most one 0. Input: [1,0,1,1,0]. Output: 4. */
int findMaxConsecutiveOnes2(vector<int>& nums) {
	int res = 0, k = 1, cnt = 0, left = 0; 
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] == 0) ++cnt; 
		while (cnt > k) {
			if (nums[left++] == 0) --cnt;
		}
		res = max(res, i - left + 1);
	}
	return res; 
}

/* 807. Max Increase to Keep City Skyline */
/* In a 2 dimensional array grid, each value grid[i][j] represents the height of a building located there. 
* We are allowed to increase the height of any number of buildings, by any amount 
* (the amounts can be different for different buildings). Height 0 is considered to be a building as well.  */
int maxIncreaseKeepingSkyline(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size(), res = 0; 
	vector<int> v1(m, 0), v2(n, 0);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			v1[i] = max(v1[i], grid[i][j]);
			v2[j] = max(v2[j], grid[i][j]);
		}
	}
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			res += abs(grid[i][j] - min(v1[i], v2[j]));
		}
	}
	return res; 
}

/* 849. Maximize Distance to Closest Person */
/* In a row of seats, 1 represents a person sitting in that seat, and 0 represents that the seat is empty. 
* There is at least one empty seat, and at least one person sitting. Alex wants to sit in the seat
* such that the distance between him and the closest person to him is maximized. Return that maximum 
* distance to closest person. Input: [1,0,0,0,1,0,1]. Output: 2. */
int maxDistToClosest(vector<int>& seats) {
	int n = seats.size(), j = -1, res = 0; 
	if (seats[0] == 1) j = 0; 
	for (int i = 1; i < n; ++i) {
		if (seats[i] == 1) {
			if (j == -1) res = i; 
			else res = max(res, (i - j) / 2);
		}
	}
	if (seats.back() == 0) res = max(res, n - 1 - j);
	return res;
}

/* 643. Maximum Average Subarray I */
/* Given an array consisting of n integers, find the contiguous subarray of given length k that 
* has the maximum average value. And you need to output the maximum average value. 
* Input: [1,12,-5,-6,50,3], k = 4. Output: 12.75. Explanation: Maximum average is (12-5-6+50)/4 = 51/4 = 12.75. */
double findMaxAverage(vector<int>& nums, int k) {
	int n = nums.size();
	double res = 0;
	vector<int> sums(n + 1, 0);
	for (int i = 1; i <= n; ++i) {
		sums[i] = sums[i - 1] + nums[i - 1];
	}
	res = sums[k - 1];
	for (int i = k; i < n; ++i) {
		res = max(res, (double)(sums[i] - sums[i - k])); 
	}
	return res / k; 
}

/* 644. Maximum Average Subarray II */
/* Given an array consisting of n integers, find the contiguous subarray whose length is 
* greater than or equal to k that has the maximum average value. And you need to output the 
* maximum average value. Input: [1,12,-5,-6,50,3], k = 4. Output: 12.75. */
double findMaxAverage(vector<int>& nums, int k) {
	int n = nums.size(); 
	vector<int> sums = nums; 
	for (int i = 1; i < n; ++i) {
		sums[i] = nums[i] + sums[i - 1];
	}
	double res = (double) sums[k - 1] / k;

	for (int i = k; i < n; ++i) {
		double t = sums[i];
		if (t > res * (i + 1)) res = t / (i + 1);

		for (int j = i - k; j >= 0; --j) {
			t = sums[i] - sums[j]; 
			if (t > res * (i - j)) res = t / (i - j);
		}
	}
	return res; 
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

/* 962. Maximum Width Ramp */
/* Given an array A of integers, a ramp is a tuple (i, j) for which i < j and A[i] <= A[j].  
* The width of such a ramp is j - i. Find the maximum width of a ramp in A. If one doesn't exist, 
* return 0. Input: [6,0,8,2,1,5]. Output: 4. Input: [9,8,1,0,1,9,4,0,4,1]. Output: 7. */
/* OTL -- No need to save all numbers */
int maxWidthRamp(vector<int>& A) {
	unordered_map<int, set<int>> m; 
	int n = A.size(), res = 0, mx = INT_MIN; 
	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			int d = A[j] - A[i]; 
			if (d >= 0) {
				m[d].insert(j - i); 
				mx = max(mx, j - i);
			}
		}
	}
	for (auto it : m) {
		for (auto a : it.second) {
			if (a == mx)  res = mx; 
		}
	}
	return res; 
}
/* Optimal solution - Use stack to save index in decreasing order nlogn */
int maxWidthRamp2(vector<int>& A) {
	stack<int> st;
	int n = A.size(), res = 0;
	for (int i = 0; i < n; ++i) {
		if (st.empty() || A[st.top()] > A[i])
			st.push(i);
	}
	for (int i = n - 1; i > res; --i) {
		while (!st.empty() && A[st.top()] <= A[i]) {
			auto t = st.top(); st.pop();
			res = max(res, i - t);
		}
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

/* 665. Non-decreasing Array */
/* Given an array with n integers, your task is to check if it could become 
 * non-decreasing by modifying at most 1 element. Input: [4,2,3]. Output: True. */
bool checkPossibility(vector<int>& nums) {
	int cnt = 1; 
	for (int i = 1; i < nums.size(); ++i) {
		if (nums[i - 1] > nums[i]) {
			if (cnt == 0) return false; 
			// IMPORTANT. 
			if (i == 1 || nums[i] >= nums[i - 2]) nums[i - 1] = nums[i];
			else nums[i] = nums[i - 1];
			--cnt; 
		}
	}
	return true; 
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

/* 750. Number Of Corner Rectangles */
/* Given a grid where each entry is only 0 or 1, find the number of corner rectangles.
* A corner rectangle is 4 distinct 1s on the grid that form an axis-aligned rectangle. 
* Note that only the corners need to have the value 1. Also, all four 1s used must be distinct. */
int countCornerRectangles(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size(), res = 0; 
	for (int k = 0; k < m; ++k) {
		for (int i = k + 1; i < m; ++i) {
			int cnt = 0; 
			for (int j = 0; j < n; ++j) {
				if (grid[k][j] == 1 && grid[i][j] == 1) ++cnt;
			}
			res += cnt * (cnt - 1) / 2; // combinations
		}
	}
	return res; 
}

/* 447. Number of Boomerangs */
/* Given n points in the plane that are all pairwise distinct, a "boomerang" is a tuple of points 
* (i, j, k) such that the distance between i and j equals the distance between i and k 
* (the order of the tuple matters). Exampel:Input: [[0,0],[1,0],[2,0]]. Output: 2. */
int numberOfBoomerangs(vector<vector<int>>& points) {
	if (points.empty() || points[0].empty()) return 0;
	int res = 0, n = points.size(); 
	for (int i = 0; i < n; ++i) {
		unordered_map<int, int> m; 
		for (int j = 0; j < n; ++j) {
			if (i == j) continue; 
			int a = abs(points[i][0] - points[j][0]);
			int b = abs(points[i][1] - points[j][1]);

			++m[a * a + b * b];
		}
		for (auto it : m) {
			res += it.second * (it.second - 1); // permutations
		}
	}
	return res; 
}

/* 792. Number of Matching Subsequences */
/* Given string S and a dictionary of words words, find the number of words[i] that is 
* a subsequence of S. Example : Input: S = "abcde", words = ["a", "bb", "acd", "ace"]
* Output: 3. Explanation: There are three words in words that are a subsequence of S: "a", "acd", "ace". */
int numMatchingSubseq(string S, vector<string>& words) {
	int res = 0;
	set<string> out, pass;
	for (auto a : words) {
		if (pass.count(a) || out.count(a)) {
			if (pass.count(a)) ++res;
			continue;
		}
		if (isSubsequence(S, a)) {
			++res;
			pass.insert(a);
		}
		else {
			out.insert(a);
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

/* 370. Range Addition */
/* Assume you have an array of length n initialized with all 0's and are given k update operations.
* Each operation is represented as a triplet: [startIndex, endIndex, inc] which increments 
* each element of subarray A[startIndex ... endIndex] (startIndex and endIndex inclusive) with inc.
* Return the modified array after all k operations were executed. 
* Input: length = 5, updates = [[1,3,2],[2,4,3],[0,2,-2]]. Output: [-2,0,3,5,3] */
vector<int> getModifiedArray(int length, vector<vector<int>>& updates) {
	vector<int> res(length + 1, 0);
	for (auto a : updates) {
		res[a[0]] += a[2];
		res[a[1] + 1] -= a[2];
	}
	for (int i = 1; i < res.size(); ++i) {
		res[i] += res[i - 1];
	}
	res.pop_back();
	return res; 
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

/* 444. Sequence Reconstruction */
/* Check whether the original sequence org can be uniquely reconstructed from the sequences in seqs. The org sequence is a 
* permutation of the integers from 1 to n, with 1 ≤ n ≤ 104. Reconstruction means building a shortest common 
* supersequence of the sequences in seqs (i.e., a shortest sequence so that all sequences in seqs are 
* subsequences of it). Determine whether there is only one sequence that can be reconstructed from seqs 
* and it is the org sequence. */
bool sequenceReconstruction(vector<int>& org, vector<vector<int>>& seqs) {
	if (seqs.empty()) return false;
	int n = org.size(), togo = n - 1;
	vector<int> pos(n + 1, 0), visited(n + 1, 0);
	for (int i = 0; i < n; ++i) pos[org[i]] = i;
	bool exist = false; // For "seqs" are empty cases.

	for (auto s : seqs) {
		for (int i = 0; i < s.size(); ++i) {
			exist = true;
			if (s[i] <= 0 || s[i] > n) return false;
			if (i == 0) continue;
			int pre = s[i - 1], cur = s[i];

			if (pos[pre] >= pos[cur]) return false;
			if (visited[cur] == 0 && pos[pre] + 1 == pos[cur]) {
				visited[cur] = 1;
				--togo;
			}
		}
	}
	return togo == 0 && exist == true;
}

/* 821. Shortest Distance to a Character */
/* Given a string S and a character C, return an array of integers representing the shortest distance 
* from the character C in the string. Input: S = "loveleetcode", C = 'e'. 
* Output: [3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0] */
vector<int> shortestToChar(string S, char C) {
	int n = S.size(), pos = -n;
	vector<int> res (n, n); 
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

/* 243. Shortest Word Distance */
/* Given a list of words and two words word1 and word2, return the shortest distance between these 
* two words in the list. Assume that words = ["practice", "makes", "perfect", "coding", "makes"].
* Input: word1 = “coding”, word2 = “practice”. Output: 3. */
int shortestDistance(vector<string>& words, string word1, string word2) {
	int res = INT_MAX;
	vector<int> idx1, idx2;
	for (int i = 0; i < words.size(); ++i) {
		if (words[i] == word1) idx1.push_back(i);
		else if (words[i] == word2) idx2.push_back(i);
	}
	for (int i = 0; i < idx1.size(); ++i) {
		for (int j = 0; j < idx2.size(); ++j) {
			res = min(res, abs(idx1[i] - idx2[j]));
		}
	}
	return res;
}

/* 360. Sort Transformed Array */
/* Given a sorted array of integers nums and integer values a, b and c. Apply a quadratic function 
* of the form f(x) = ax2 + bx + c to each element x in the array. The returned array must be in sorted order.
* Expected time complexity: O(n). */
int cal(int x, int a, int b, int c) {
	return a * x * x + b * x + c;
}

vector<int> sortTransformedArray(vector<int>& nums, int a, int b, int c) {
	int n = nums.size(), i = 0, j = n - 1;
	int idx = a >= 0 ? n - 1 : 0;
	vector<int> res(n, 0);

	while (i <= j) {
		if (a >= 0) {
			// 如果a>0，则抛物线开口朝上，那么两端的值比中间的大，从后往前更新res.
			res[idx--] = cal(nums[i], a, b, c) > cal(nums[j], a, b, c) ? cal(nums[i++], a, b, c) : cal(nums[j--], a, b, c);
		}
		else { // 如果a<0，则抛物线开口朝下，则两端的值比中间的小，从前往后更新res.
			res[idx++] = cal(nums[i], a, b, c) > cal(nums[j], a, b, c) ? cal(nums[j--], a, b, c) : cal(nums[i++], a, b, c);
		}
	}
	return res;
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

/* 977. Squares of a Sorted Array */
/* Given an array of integers A sorted in non-decreasing order, return an array of the squares of 
* each number, also in sorted non-decreasing order. Example 1: Input:[-4,-1,0,3,10]. Output:[0,1,9,16,100]. */
vector<int> sortedSquares(vector<int>& A) {
	int n = A.size(), i = 0, j = n - 1;
	vector<int> res;
	while (i <= j) {
		res.insert(res.begin(), max(A[i] * A[i], A[j] * A[j]));
		A[i] * A[i] < A[j] * A[j] ? --j : ++i;
	}
	return res;
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

/* 414. Third Maximum Number */
/* Given a non-empty array of integers, return the third maximum number in this array. 
* If it does not exist, return the maximum number. The time complexity must be in O(n).
* Example 1: Input: [3, 2, 1]. Output: 1. */
int thirdMax(vector<int>& nums) {
	long long mx1 = LONG_MIN, mx2 = LONG_MIN, mx3 = LONG_MIN;
	for (auto a : nums) {
		if (a > mx1) {
			mx3 = mx2;
			mx2 = mx1;
			mx1 = a;
		}
		else if (a > mx2 && a < mx1) {
			mx3 = mx2;
			mx2 = a;
		}
		else if (a > mx3 && a < mx2) {
			mx3 = a;
		}
	}
	return (mx3 == LONG_MIN || mx3 == mx2) ? mx1 : mx3;
}

/* 766. Toeplitz Matrix */
/* A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same element.
* Now given an M x N matrix, return True if and only if the matrix is Toeplitz. */
bool isToeplitzMatrix(vector<vector<int>>& matrix) {
	for (int i = 0; i < matrix.size() - 1; ++i) {
		for (int j = 0; j < matrix[0].size() - 1; ++j) {
			if (matrix[i][j] != matrix[i + 1][j + 1]) return false;
		}
	}
	return true;
}


// =============================================================

// ====================== 17. OTHERS (MATH) ====================
/* 343. Integer Break */
int integerBreak(int n) {
	int res = 1;
	if (n <= 2) return 1;
	if (n == 3) return n - 1;
	if (n == 4) return n;

	while (n > 4) {
		res *= 3;
		n -= 3;
	}
	return n * res;
}

/* 397. Integer Replacement */
/* Given a positive integer n and you can do operations as follow:
* If n is even, replace n with n/2.
* If n is odd, you can replace n with either n + 1 or n - 1.
* What is the minimum number of replacements needed for n to become 1? */
int integerReplacement(int n) {
	if (n <= 1) return 0; 
	while (n > 1) {
		if (n % 2 == 0) return 1 + integerReplacement(n / 2);
		else return 2 + min(integerReplacement((n - 1) / 2 + 1), integerReplacement((n - 2) / 2)); 
	}
	return -1; 
}

/* 812. Largest Triangle Area */
/* You have a list of points in the plane. Return the area of
* the largest triangle that can be formed by any 3 of the points. */
double largestTriangleArea(vector<vector<int>>& points) {
	double res = 0; 
	for (auto a : points) {
		for (auto b : points) {
			for (auto c : points) {
				int x1 = a[0], y1 = a[1];	
				int x2 = b[0], y2 = b[1]; 
				int x3 = c[0], y3 = c[1]; 
				res = max(res, 0.5 * (x1 * y2 + x2 * y3 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3));
			}
		}
	}
	return res; 
}

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

/* 356. Line Reflection */
/* Given n points on a 2D plane, find if there is such a line parallel 
 * to y-axis that reflect the given points. */
bool isReflected(vector<vector<int>>& points) {
	int mn = INT_MAX, mx = INT_MIN; 
	unordered_map<int, unordered_set<int>> m;
	for (auto a : points) {
		m[a[0]].insert(a[1]);
		mn = min(mn, a[0]);
		mx = max(mx, a[0]);
	}
	
	double x0 = (double) (mn + mx) / 2; 
	for (auto a : points) {
		int t = 2 * x0 - a[0]; 
		if (!m.count(t) || !m[t].count(a[1])) return false; 
	}
	return true; 
}

/* 858. Mirror Reflection */
/* There is a special square room with mirrors on each of the four walls. 
* Except for the southwest corner, there are receptors on each of the remaining corners,
* numbered 0, 1, and 2. The square room has walls of length p, and a laser ray from the 
* southwest corner first meets the east wall at a distance q from the 0th receptor.
* Return the number of the receptor that the ray meets first.  
* (It is guaranteed that the ray will meet a receptor eventually.) 
* LOGIC: Divide p,q by 2 until at least one odd.
* If p = odd, q = even: return 0, If p = even, q = odd: return 2, If p = odd, q = odd: return 1 */
int mirrorReflection(int p, int q) {
	while (p % 2 == 0 && q % 2 == 0) { p >>= 1; q >>= 1; }
	return 1 - p % 2 + q % 2;
}

/* 400. Nth Digit */
/* Find the nth digit of the infinite integer sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... */
int findNthDigit(int n) {
	long long cnt = 9, len = 1, start = 1;

	while (n > cnt * len) {
		n -= cnt * len;
		cnt *= 10;
		start *= 10;
		++len;
	}

	start += (n - 1) / len;
	string s = to_string(start);
	return s[(n - 1) % len] - '0';
}

/* 326. Power of Three */
/* Given an integer, write a function to determine if it is a power of three. Input: 27. Output: true. */
bool isPowerOfThree(int n) {
	if (n <= 0) return false;
	while (n > 0 && n % 3 == 0) {
		n /= 3;
	}
	return n == 1;
}

/* 780. Reaching Points */
/* A move consists of taking a point (x, y) and transforming it to either (x, x+y) or (x+y, y).
* Given a starting point (sx, sy) and a target point (tx, ty), return True if and only if 
* a sequence of moves exists to transform the point (sx, sy) to (tx, ty). Otherwise, return False.
* Examples: Input: sx = 1, sy = 1, tx = 3, ty = 5. Output: True.
* 快速累加的高效方法是乘法，但要知道需要累加的个数，就需要用除法来计算，其实我们对累加的个数也不那么感兴趣，
* 而是对余数感兴趣，那么求余运算就是很高效的方法。 我们的目标是将 tx 和 ty 分别缩小到 sx 和 sy，不可能一步就缩小到位，
* 那么这肯定是一个循环，终止条件是 tx 和 ty 中任意一个小于了 sx 和 sy，在循环内部，想要缩小 tx 或 ty，
* 先缩小两者中较大的那个. */
bool reachingPoints(int sx, int sy, int tx, int ty) {
	while (tx >= sx && ty >= sy) {
		if (tx > ty) {
			// 因为此时 ty 不能改变了，只能缩小 tx
			if (ty == sy) return (tx - sx) % ty == 0;
			tx %= ty; 
		}
		else {
			if (tx == sx) return (ty - sy) % tx == 0;
			ty %= tx;
		}
	}
	return tx == sx && ty == sy; 
}

/* 660. Remove 9 */
/* Start from integer 1, remove any integer that contains 9 such as 9, 19, 29...
* So now, you will have a new integer sequence: 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, ...
* Given a positive integer n, you need to return the n-th integer after removing. 
* Note that 1 will be the first integer. Input: 9. Output: 10. Hint: n will not exceed 9 x 10^8.
* 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 
* 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, ... , 88, 100, 101, ... 108. 
* 可以发现，8的下一位就是10了，18的下一位是20，88的下一位是100，实际上这就是九进制的数字的规律，
* 那么这道题就变成了将十进制数n转为九进制数. 就每次对9取余，然后乘以base，n每次自除以9，base 每次扩大10倍. */
int newInteger(int n) {
	long res = 0, base = 1; 
	while (n > 0) {
		res = n % 9 * base; 
		n /= 9; 
		base *= 10; 
	}
	return res; 
}

/* 800. Similar RGB Color */
/* In the following, every capital letter represents some hexadecimal digit from 0 to f. 
* Given the color "#ABCDEF", return a 7 character color that is most similar to #ABCDEF, 
* and has a shorthand (that is, it can be represented as some "#XYZ". Input: color = "#09f166"
* Output: "#11ee66".*/
string similarRGBHelper(string str) {
	string dict = "0123456789abcdef";
	// convert hexadecimal string to 10 base number
	int num = stoi(str, nullptr, 16);
	// convert 10 base number to hexadecimal digit with closest diff
	/* "09"来举例，这个数字可以变成"11"或者"00"，十六进制数"11"对应的十进制数是17，跟"09"相差了8，
	* 而十六进制  数"00"对应的十进制数是0，跟"09"相差了9，显然我们选择"11"会好一些。所以我们的临界点是"8"，
	* 如果个位上的数字大于"8"，那么十位上的数就加1。 */
	int idx = num / 17 + (num % 17 > 8 ? 1 : 0);
	return string(2, dict[idx]);
}

string similarRGB(string color) {
	return "#" + similarRGBHelper(color.substr(1, 2)) + similarRGBHelper(color.substr(3, 2)) + similarRGBHelper(color.substr(5, 2));
}

/* 1015. Smallest Integer Divisible by K */
/* Given a positive integer K, you need find the smallest positive integer N such that N is divisible by K,
* and N only contains the digit 1. Return the length of N.  If there is no such N, return -1. 
* LOGIC: Let's say the final result has N digits of 1. If N exist, N <= K, just do a brute force check.
* Also if K % 2 == 0, return -1, because 111....11 can't be even.
* Also if K % 5 == 0, return -1, because 111....11 can't end with 0 or 5.*/
int smallestRepunitDivByK(int K) {
	for (int r = 0, N = 1; N <= K; ++N)
		if ((r = (r * 10 + 1) % K) == 0)
			return N;
	return -1;
}

/* 640. Solve the Equation */
/* Solve a given equation and return the value of x in the form of string "x=#value". The equation contains 
* only '+', '-' operation, the variable x and its coefficient. If there is no solution for the equation, 
* return "No solution". If there are infinite solutions for the equation, return "Infinite solutions".
* If there is exactly one solution for the equation, we ensure that the value of x is an integer. */
string solveEquation(string equation) {
	string res("");
	int i = 0, n = equation.size(), flag = 1; 
	int para = 0, xpara = 0;

	while (i < n) {
		int sign = 1, t = 0; 
		if (equation[i] == '=') {
			flag = -1; 
			++i;
		}
		if (equation[i] == '-') {
			sign = -1; 
			++i;
		}
		if (equation[i] == '+') {
			sign = 1; 
			++i;
		}
		if (isdigit(equation[i])) {
			while (i < n && isdigit(equation[i])) {
				t = t * 10 + (equation[i++] - '0');
			}
			if (i < n && equation[i] == 'x') {
				xpara += flag * sign * t; 
				++i; 
			}
			else {
				para += flag * sign * t; 
			}
		}
		else {
			xpara += flag * sign; 
			++i;
		}
		if (para == 0 && xpara == 0) {
			res = "Infinite solutions";
		}
		else if (xpara == 0) {
			res = "No solution";
		}
		else {
			res = "x=" + to_string(para / xpara * -1);
		}
	}
	return res; 
}

/* 777. Swap Adjacent in LR String */
/* In a string composed of 'L', 'R', and 'X' characters, like "RXXLRXRXL", 
* a move consists of either replacing one occurrence of "XL" with "LX", 
* or replacing one occurrence of "RX" with "XR". Given the starting string start 
* and the ending string end, return True if and only if there exists a sequence 
* of moves to transform one string to the other. Example:
* Input: start = "RXXLRXRXL", end = "XRLXXRRLX". Output: True. */
bool canTransform(string start, string end) {
	int m = start.size(), n = end.size(), i = 0, j = 0;
	while (i < m && j < n) {
		while (i < m && start[i] == 'X') ++i;
		while (j < n && end[j] == 'X') ++j;

		if (start[i] != end[j]) return false;
		if ((end[j] == 'L' && i < j) || (start[i] == 'R' && i > j)) return false;
		++i;
		++j;
	}
	return true;
}

// ====================== TO DO LIST =========================
/* 471. Encode String with Shortest Length */
/* Given a non-empty string, encode the string such that its
* encoded length is the shortest. The encoding rule is: k[encoded_string],
* where the encoded_string inside the square brackets is being
* repeated exactly k times. Input: "aaaaa". Output: "5[a]". */
/*
string encode(string s) {
}
*/



// ====================== QUESTIONS ==========================
/* 932. Beautiful Array */


// ===========================================================

int main() {
	cout << "unit test" << endl; 
}

