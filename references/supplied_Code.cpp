#include <bits/stdc++.h>
#define ll long long
#define ull unsigned long long
#define int long long
#define all(x) x.begin(),x.end()
#define yes cout<<"YES\n";
#define no cout<<"NO\n";
#define ln "\n"
#define MOD 1000000007
#define fi(ii, n) for (ll ii = 1; ii <= n; ii++)
#define fd(ii,n) for (ll ii=n; ii>=0; ii--)
#define vi vector<int>
#define vll vector<ll>
#define vvi vector<vector<int>>
#define pii pair<int,int>
#define pb push_back
using namespace std;


void solve() {
int n,k;cin>>n>>k;

vi v(n),f(201,0);
for(auto &x:v)cin>>x;

for(auto &x:v)f[x]++;

int take=0,res = 0;

for(int i=0;i<=n;i++){
    take += f[i];
    if(f[i]==0 || take<k){
        break;
    }
    take -=k;
    ++res;
}
cout<<res<<ln;
}

signed main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t = 1;
    cin >> t;
    while (t--) solve();
    return 0;
}
