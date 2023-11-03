// 1. Binomial Coeffecients:

#define NCR
struct COMB {
  std::vector<long long> fact, invfact;
  int mod;
  void init(int N, int M) {
    mod = M;
    fact.assign(N + 5, 0);
    invfact.assign(N + 5, 0);
    int p = mod;   //should be prime
    fact[0] = 1;
    int i;
    for(i = 1 ; i < N ; i++){
      fact[i] = (1LL * i * fact[i-1]) % p;
    }
    i--;
    invfact[i] = bpow(fact[i] , p - 2);
    for(i-- ; i >= 0 ; i--){
      invfact[i] = (1LL * invfact[i+1] * (i+1)) % p;
    }
  }
  long long bpow(long long n , long long x) {
    long long res = 1;
    n = n % mod;
    while (x > 0) {
      if (x % 2 == 1) res = (res * n) % mod;
      n = (n * n) % mod;
      x /= 2;
    }
    return res; 
  }
  int ncr(int n , int r) {   
     if (r > n || n < 0 || r < 0) return 0;
     return fact[n] * invfact[r] % mod * invfact[n-r] % mod;
  }   
};
constexpr int mod = 1E9 + 7;
COMB q;
// q.init(N, mod); // inside main

// 2. LCA + Binary Lifiting

struct LCA {
    int n, l, timer;
    vector<vector<int>> adj, up;
    vector<int> tin, tout, depth;
    LCA (vector<vector<int>> _adj, int root) : adj(_adj) {
        n = (int) adj.size();
        l = (int) ceil(log2(n)) + 1;
        timer = 0;
        tin.resize(n);
        tout.resize(n);
        depth.resize(n);
        up.assign(n, vector<int>(l));
        dfs(root, root, 0);
    }
    void dfs (int node, int prev, int d) {
        tin[node] = ++timer;
        up[node][0] = prev;
        depth[node] = d;
        for(int i = 1; i < l; i++) { up[node][i] = up[up[node][i - 1]][i - 1]; }
        for(auto next : adj[node]) { if(next != prev) dfs(next, node, d + 1); }
        tout[node] = ++timer;
    }
    bool is_ancestor (int x, int y) { return tin[x] <= tin[y] && tout[x] >= tout[y]; }
    int query (int u, int v) {
        if(is_ancestor(u, v)) return u;
        if(is_ancestor(v, u)) return v;
        for(int i = l - 1; i >= 0; i--) { if(!is_ancestor(up[u][i], v)) u = up[u][i]; }
        return up[u][0];
    }
    int jump (int x, int k) {
        for(int i = l - 1; i >= 0; i--) if((k>>i) & 1) x = up[x][i];
        return x;
    }
    int parent (int x) {
        return up[x][0];
    }
    int distance (int x, int y) {
        int z = query(x, y);
        return depth[x] + depth[y] - 2 * depth[z];
    }
};

struct LCA {
  int n;
  vector<vector<int>> dp;
  vector<int> d;

  void init(vector<int> parent, vector<int> depth) {
    n = parent.size();
    d = vector<int>(depth);
    dp.assign(n, vector<int>(23, 0));
    for (int i = 2; i < n; i += 1) {
      dp[i][0] = parent[i];
    }
    for (int k = 1; k < 23; k += 1) {
      for (int i = 1; i < n; i += 1) {
        dp[i][k] = dp[dp[i][k - 1]][k - 1];
      }
    }
  }

  int query(int node, int level) {
    bitset<22> l(level);
    for (int i = 0; i < 22; i += 1) {
      if (l[i]) {
        node = dp[node][i];
      }
    }
    return node;
  }

  int lca(int node_a, int node_b) {
    if (d[node_a] != d[node_b]) {
      node_a = query(node_a, max(d[node_a] - d[node_b], 0));
      node_b = query(node_b, max(d[node_b] - d[node_a], 0));
    }

    if (node_a == node_b) {return node_a;}
    else {
      for (int i = 21; i >= 0; i -= 1) {
        if (dp[node_a][i] != dp[node_b][i]) {
          node_a = dp[node_a][i];
          node_b = dp[node_b][i];
        }
      }
      return dp[node_a][0];
    }
  }
};

// 3. DSU

#define DISJOINT_SET_UNION
struct DSU {
    std::vector<int> parent , siz;
    DSU(int n){
      parent.resize(n + 1); 
      siz.resize(n + 1);
      for(int i = 0; i < n + 1; i++) {
         parent[i] = i; 
         siz[i] = 1;
      }
    }
    int leader (int i) {
      if(parent[parent[i]] != parent[i]) {
         parent[i] = leader(parent[i]);
      }
      return parent[i];
    }
    bool same (int x, int y) { 
      return leader(x) == leader(y); 
    }
    void merge (int a, int b) {
      a = leader(a), b = leader(b);
      if(a == b) return;
      if(siz[b] >= siz[a]) swap(a, b);
      siz[a] += siz[b];
      parent[b] = a;
      return;
    }
    int size(int x) { 
      return siz[leader(x)]; 
    }
}; 

// 4. Djikstra
int const inf = 1e15;
void dijkstra(int source,vector<int>& d,vector<vector<pair<int,int>>>& adj) {
    set<pair<int,int>> s;
    d[source] = 0;
    s.insert({0, source});
    while (!s.empty()) {
        int v = s.begin()->second;  //choosing min unmarked v
        s.erase(s.begin());
        // performing relaxation
        for (auto i: adj[v]) {
            int to = i.first;
            int wt = i.second;
            if (d[to] >= inf || d[v] + wt < d[to]) {
                s.erase({d[to], to});
                d[to] = d[v] + wt;
                s.insert({d[to], to});
            }
        }
    }
}
// 5. PBDS

#include <ext/pb_ds/tree_policy.hpp>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;

template<typename T>
using O_S = tree<T,  null_type,  less<T>,  rb_tree_tag,  tree_order_statistics_node_update>;
template<typename T>
using O_MS = tree<T,  null_type,  less_equal<T>,  rb_tree_tag,  tree_order_statistics_node_update>;

template<typename T>
struct ordered_multiset {
  O_MS<T> s;
  void insert (T x) {
    s.insert(x);
  }
  bool exist (T x) {
    if((s.upper_bound(x)) == s.end()) {
      return 0;
    }
    return ((*s.upper_bound(x)) == x);
  }
  void erase (T x) {
    if (exist(x)) {
      s.erase(s.upper_bound(x));
    }
  }
  int firstIdx (T x) {
    if(!exist(x)) {
        return -1;
    }
    return (s.order_of_key(x));
  }
  T value (int idx) { 
    return (*s.find_by_order(idx));
  }
  int lastIdx (T x) { 
    if(!exist(x)){
        return -1;
    }
    if(value((int)s.size() - 1) == x){
        return (int)(s.size()) - 1;
    }
    return firstIdx(*s.lower_bound(x)) - 1;
  }
  int count(T x) { 
    if(!exist(x)) {
      return 0;
    }
    return lastIdx(x) - firstIdx(x) + 1;
  }
  int size () {
    return (int) s.size();
  }
  void clear() {
    s.clear();
  } 
  void print() {
    for (auto z: s) {
      cout << z << ' ';
    }
    cout << '\n';
  }
};

template<typename T>
struct ordered_set {
  O_S<T> s;
  void insert (T x) {
    s.insert(x);
  }
  bool exist (T x) {
    int f = s.order_of_key(x);
    if(f == (int) s.size()) {
      return 0;
    }
    return (value(f) == x);
  }
  void erase (T x) {
    if (exist(x)) {
      s.erase(x);
    }
  }
  T value (int idx) { 
    return (*s.find_by_order(idx));
  }
  int less (T x) { // how many elements strictly less than 'x'
    int f = s.order_of_key(x);
    return f;
  }
  int size () {
    return (int) s.size();
  }
  void clear() {
    s.clear();
  } 
  void print() {
    for (auto z: s) {
      cout << z << ' ';
    }
    cout << '\n';
  }
};

// use as 'ordered_set<int> s' inside main.

// 6. KMP

/* 
       - src : https://cp-algorithms.com/string/prefix-function.html
       - pi[i] = longest proper prefix of s[0....i] which is also a suffix of this substring.
       - for kmp we use (s + '#' + t)
*/
// made as lambda

auto prefix_function = [&] (string s) {  // works in O(N)
    int n = (int) s.size();
    vector<int> pi(n , 0);
    for(int i = 1; i < n; i++) {
        int j = pi[i - 1];
        while(j > 0 and s[i] != s[j]) {
            j = pi[j - 1];
        }
        if(s[i] == s[j]) j++;
        pi[i] = j;
    }
    return pi;
};

// 7. Segment Tree

struct SGT {   //segtree implementation by geothermal
  std::vector<int> seg; 
  int SZ;
  SGT() {};
  int combine (int a, int b) { return max(a, b); } // change to sum, max, min
  void build (vector<int> val) {
    SZ = (int) val.size();
    seg.resize(2 * SZ + 1); // put identity here
    for(int i = SZ; i < 2 * SZ; ++i) {
      seg[i] = val[i - SZ];
    }
    for(int i = SZ - 1; i > 0; --i) seg[i] = combine(seg[i << 1], seg[(i << 1) | 1]); 
  }
  void update (int p, int v) {
    p += SZ; seg[p] = v;  // for ADDING to point, seg[p] += v & for SETTING, seg[p] = v.
    for(int i = p; i > 1; i >>= 1) seg[i >> 1] = combine(seg[i], seg[i ^ 1]); 
  }
  int get (int idx) {
    return seg[idx + SZ];
  }
  int query (int l, int r) {
    int ans = 0;  // set this to identity
    if (l > r) return ans;
    l = max(0LL, l);
    r = min(SZ - 1, r);
    for(l += SZ, r += SZ + 1; l < r; l >>= 1, r >>= 1) {
      if (l & 1) ans = combine(ans, seg[l++]); 
      if (r & 1) ans = combine(ans, seg[--r]);  
    }
    return ans;
  }
};

// #include<bits/stdc++.h>
// using namespace std;
// #define int long long int

int add(int x, int y) {return x + y;}
int id() {return 0;}
template<typename T, T (*op)(T, T), T (*id)()>
class SegmentTree{
  T sz;
  vector<T> s;

  void init(T n) {
    sz = (n == 1 ? n : 1 << (32 - __builtin_clz(n - 1))) << 1;
    --sz;
    s.assign(sz, id());
  }

  public:
  void build(vector<T> &a) {
    int n = a.size();
    init(n);
    int ofs = (1 << (__builtin_popcount(sz & (sz - 1)))) - 1;
    for (int i = 0; i < n; i += 1) {
     s[i + ofs] = a[i]; 
    }

    for (int i = ofs - 1; i >= 0; i -= 1) {
      s[i] = op(s[(i << 1) + 1], s[(i << 1) + 2]);
    }
  }

  T query(T l, T r){
    return query(l, r, 0, 0, (sz >> 1) + 1);
  }

  T query(T l, T r, T x, T left, T right) {
    if (l >= right || r <= left) {
      return id();
    }
    else if (l <= left && right <= r) {
      return s[x];
    }
    else {
      T mid = (left + right) >> 1;
      return op(query(l, r, (x << 1) + 1, left, mid), query(l, r, (x << 1) + 2, mid, right));
    }
  }

  void update(T index, T value) {
    update(index, value, 0, 0, (sz >> 1) + 1);
  }

  void update(T index, T value, T x, T left, T right) {
    if (right - left == 1) {
      s[x] = value;
      return;
    }
    T mid = (left + right) >> 1;
    if (index < mid) {
      update(index, value, (x << 1) + 1, left, mid);
    }
    else {
      update(index, value, (x << 1) + 2, mid, right);
    }
    s[x] = op(s[(x << 1) + 1], s[(x << 1) + 2]);
  }
};


// signed main() {
//   int n, m;
//   cin >> n >> m;
//   vector<int> a(n);
//   for (int i = 0; i < n; i += 1) {
//     cin >> a[i];
//   }

//   SegmentTree<int, add, id> S;
//   S.build(a);

//   while(m--) {
//     int l, r;
//     cin >> l >> r;
//     cout << S.query(l, r) << endl;
//   }

// }

// 8. Sieve Prime

// use this till N around 1E7
// O(n * log(log(n)))
// number of primes less than N are around (N / log(N))
// can be checked using primes.size()

#define SIEVE   
int is_prime[N];
vector<int> primes;
int spf[N];
void sieve() {
    for(int i = 1; i < N; i++) {
        spf[i] = i;
    }
    for(int i = 2; i < N; i++) {
        if (is_prime[i] == 0) {
            primes.push_back(i);
            for(auto j = 1ll * i * i ; j < N ; j += i){
                is_prime[j] = 1;
                if (spf[j] == j) spf[j] = i;
            }
        }
        is_prime[i] ^= 1;
    }
}
vector<int> factorise(int n){  // O(logN) , works till n ranging 1e7.
    vector<int> ans;
    while(n != 1){
        ans.push_back(spf[n]);
        n = n / spf[n];
    }
    return ans;     // ans vector is always sorted.
}

auto primeFactors = [&] (int n) {    //O (sqrtN)
    std::vector<int> factors;
    for (int i = 2; i * i <= n; i++) { 
        while (n % i == 0) {
            factors.push_back(i);
            n = n / i;
        }
    }
    if (n > 2) factors.push_back(n); // if n is a prime greater than 2  
    return factors;  
};

// 9. Sparse Table

// use only for O(1) range max/min/gcd queries.
// use segtree/fenwick for rest of queries O(logN).
// add gcd query function when needed.

struct SparseTable {
  vector<vector<int>> table_max;
  vector<vector<int>> table_min;
  SparseTable(vector<int>& arr) {
    int n = (int) arr.size();
    int N1 = n + 5;
    int N2 = (int) lg(N1) + 2;
    table_max.assign(N1, vector<int>(N2));
    table_min.assign(N1, vector<int>(N2));
    for (int i = 0; i < n; i++) {
      table_max[i][0] = arr[i];
      table_min[i][0] = arr[i];
    }
    for (int j = 1 ; j <= lg(n) ; j++) {
      for (int i = 0 ; i <= n - (1 << j) ; i++) {
        table_max[i][j] = max( table_max[i][j - 1] , table_max[i + (1 << (j - 1))][j - 1] );
        table_min[i][j] = min( table_min[i][j - 1] , table_min[i + (1 << (j - 1))][j - 1] );
      }
    }
  }
  int amax(int L, int R) {
    int j = (int)lg(R - L + 1);
    return max(table_max[L][j], table_max[R - (1 << j) + 1][j]);
  }
  int amin(int L, int R) {
    int j = (int)lg(R - L + 1);
    return min(table_min[L][j], table_min[R - (1 << j) + 1][j]);
  }
  int lg(int x) const { return 31 - __builtin_clz(x); }
};

// 10. String Hashing

// credits : demoraliser
// look where to declare b1, b2. declare inside main for same hashing among all structs (IMP)
// sample submission : https://codeforces.com/contest/271/submission/175524714

struct Hash {   
    int n;
    int b1, b2;
    vector<int> pb1, pb2;
    vector<int> pre_1, pre_2;
    int MOD1, MOD2;

    Hash(string s, int _mod1 = 101777101, int _mod2 = 999999929) {
        n = (int) s.size();
        std::mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
        auto dist=uniform_int_distribution<int>(1000,5000);
        b1 = dist(rng);
        b2 = dist(rng);
        
        MOD1 = _mod1 , MOD2 = _mod2;
        pb1.assign(n + 1, 0); pb2.assign(n + 1, 0);
        pre_1.assign(n + 1, 0); pre_2.assign(n + 1, 0);  

        pb1[0] = pb2[0] = 1;
        for(int i = 1; i < n + 1; i++) {
            pb1[i] = pb1[i-1] * 1ll * b1 % MOD1;
            pb2[i] = pb2[i-1] * 1ll * b2 % MOD2;
        }
        
        for(int i = 1; i < n + 1; i++) {
            pre_1[i] = (pre_1[i - 1] + s[i - 1] * 1ll * pb1[i - 1]) % MOD1;
            pre_2[i] = (pre_2[i - 1] + s[i - 1] * 1ll * pb2[i - 1]) % MOD2;
        }
    }

    pair<int,int> code (int l, int r) {   // [l, r] included end
        int H1 = (pre_1[r + 1] + MOD1 - pre_1[l]) * 1ll * pb1[n - r] % MOD1;
        int H2 = (pre_2[r + 1] + MOD2 - pre_2[l]) * 1ll * pb2[n - r] % MOD2;
        return {H1, H2};
    }
}; 

// 11. Mint template
const int32_t mod = 1E9 + 7;

int norm(int x) {
  if (x < 0) {
    x += mod;
  }
  if (x >= mod) {
    x -= mod;
  }
  return x;
}
template<class T>
T power(T a, long long b) {
  T res = 1;
  for (; b; b /= 2, a *= a) {
    if (b % 2) {
    res *= a;
    }
  }
  return res;
}
struct Mint {
  int x;
  Mint(int x = 0) : x(norm(x)) {}
  // Mint(long long x) : x(norm(x % mod)) {}   // uncomment when there is no int = ll
  int val() const {
    return x;
  }
  Mint operator-() const {
    return Mint(norm(mod - x));
  }
  Mint inv() const {
    assert(x != 0);
    return power(*this, mod - 2);
  }
  Mint &operator*=(const Mint &rhs) {
    x = (long long) x * rhs.x % mod;
    return *this;
  }
  Mint &operator+=(const Mint &rhs) {
    x = norm(x + rhs.x);
    return *this;
  }
  Mint &operator-=(const Mint &rhs) {
    x = norm(x - rhs.x);
    return *this;
  }
  Mint &operator/=(const Mint &rhs) {
    return *this *= rhs.inv();
  }
  friend Mint operator*(const Mint &lhs, const Mint &rhs) {
    Mint res = lhs;
    res *= rhs;
    return res;
  }
  friend Mint operator+(const Mint &lhs, const Mint &rhs) {
    Mint res = lhs;
    res += rhs;
    return res;
  }
  friend Mint operator-(const Mint &lhs, const Mint &rhs) {
    Mint res = lhs;
    res -= rhs;
    return res;
  }
  friend Mint operator/(const Mint &lhs, const Mint &rhs) {
    Mint res = lhs;
    res /= rhs;
    return res;
  }
  friend std::istream &operator>>(std::istream &is, Mint &a) {
    long long v;
    is >> v;
    a = Mint(v);
    return is;
  }
  friend std::ostream &operator<<(std::ostream &os, const Mint &a) {
    return os << a.val();
  }
};

std::vector<Mint> fact, invfact;
void Compute_facts (int N) {
  fact.resize(N + 5);
  invfact.resize(N + 5);
  fact[0] = 1;
  for (int i = 1; i <= N; i++) {
    fact[i] = fact[i - 1] * i;
  }
  invfact[N] = fact[N].inv();
  for (int i = N; i > 0; i--) {
    invfact[i - 1] = invfact[i] * i;
  }
}

Mint ncr(int n, int r) {
  Mint ans = fact[n] * invfact[n - r] * invfact[r];
  return ans;
}


#include <bits/stdc++.h>

using namespace std;
typedef long long int ll;
typedef vector<ll> vi;
typedef pair<ll, ll> pi;

#define FAST ios::sync_with_stdio(0), cin.tie(0)
#define rep(i,a,b) for(ll i=a;i<b;i++)
#define per(i,a,b) for(ll i=a;i>b;i--)
#define endl '\n'

const ll MAX = INT_MAX;
const ll MIN = INT_MIN;
const ll MOD = 1e9 + 7;

ll ceil(ll a, ll b) {
    return (a + b - 1) / b; 
}


void dfsHighPoints(ll v, vector<ll>* adj, vector<bool>& visited, vector<ll>& hps, vector<ll>& level, vector<ll>& parent){
    visited[v] = true;

    for (auto x: adj[v]){
        if (x!=parent[v]){

            // back edge
            if (visited[x]){
                hps[v] = min(hps[v], level[x]);
            }
            // forward edge
            else{
                level[x] = level[v] + 1;
                parent[x] = v;
                dfsHighPoints(x, adj, visited, hps, level, parent);
                hps[v] = min(hps[v], hps[x]);
            }
        }
    }
}


void dfsDp(ll v, vector<ll>* adj, vector<ll>& dp, vector<bool>& visited, vector<ll>& level, vector<ll>& parent){
    visited[v] = true;

    for (auto x: adj[v]){
        if (x!=parent[v]){

            if (visited[x]){
                // Back edge going up from v
                if (level[x] < level[v]){
                    dp[v]++;
                }
                // Back edge going down from v
                else{
                    dp[v]--;
                }
            }
            else{
                level[x] = level[v]+1;
                parent[x] = v;

                dfsDp(x, adj, dp, visited, level, parent);

                // Back edge going up from child of v
                dp[v] += dp[x];

            }


        }
    }
}

void solve(){
    ll n, m;
    cin >> n >> m;

    vi adj[n];

    while(m--){
        ll a, b;
        cin >> a >> b;

        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    vector<bool> visited(n, false);
    vector<ll> parent(n, -1);
    vector<ll> level(n, 0);
    vector<ll> hps(n, INT_MAX);
    vector<ll> dp(n, 0);


    // Approach 1: DP on High Points
    dfsHighPoints(0, adj, visited, hps, level, parent);
    rep(i,1,n){
        if (hps[i] > level[i]-1){
            cout << parent[i] << " " << i  << endl;
        }
    }

    // Approach 2: DP on Edges Only
    // dp[u] = no of back edges going up from u + no of back edges going down from u - sigma dp[v] (v -> children of u)
    dfsDp(0, adj, dp, visited, level, parent);
    rep(i,1,n){
        if (!dp[i]){
            cout << parent[i] << " " << i << endl;
        }
    }

}


int main() {
    FAST;
    ll t;
    cin >> t;
    while(t--){
        solve();
    }
    return 0;
}