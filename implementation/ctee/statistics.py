"""Explicit paired-test statistics and distribution tails; NumPy and math only.

Incomplete-beta/gamma continued fractions follow the standard mathematical
representations catalogued in NIST DLMF sections 8.17 and 8.9.
"""
import math
from itertools import combinations,product
import numpy as np


def _nonzero(x):return x if abs(x)>1e-300 else 1e-300

def _beta_fraction(a,b,x):
    total=a+b;c=1.;d=1/_nonzero(1-total*x/(a+1));h=d
    for m in range(1,501):
        twice=2*m
        aa=m*(b-m)*x/((a-1+twice)*(a+twice))
        d=1/_nonzero(1+aa*d);c=_nonzero(1+aa/c);h*=d*c
        aa=-(a+m)*(total+m)*x/((a+twice)*(a+1+twice))
        d=1/_nonzero(1+aa*d);c=_nonzero(1+aa/c);delta=d*c;h*=delta
        if abs(delta-1)<1e-13:return h
    raise ArithmeticError('Incomplete beta did not converge')

def beta_regularized(a,b,x):
    if a<=0 or b<=0 or not 0<=x<=1:raise ValueError('Invalid beta arguments')
    if x==0:return 0.
    if x==1:return 1.
    multiplier=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    value=(multiplier*_beta_fraction(a,b,x)/a if x<(a+1)/(a+b+2)
           else 1-multiplier*_beta_fraction(b,a,1-x)/b)
    return min(1.,max(0.,value))

def f_survival(f,df1,df2):
    if f<0 or df1<=0 or df2<=0:raise ValueError('Invalid F statistic')
    return beta_regularized(df2/2,df1/2,df2/(df2+df1*f))

def gamma_survival(a,x):
    if a<=0 or x<0:raise ValueError('Invalid gamma arguments')
    if x==0:return 1.
    factor=math.exp(-x+a*math.log(x)-math.lgamma(a))
    if x<a+1:
        term=total=1/a
        for n in range(1,1001):
            term*=x/(a+n);total+=term
            if abs(term)<abs(total)*1e-14:return min(1.,max(0.,1-total*factor))
    else:
        b=x+1-a;c=1e300;d=1/_nonzero(b);h=d
        for n in range(1,1001):
            an=-n*(n-a);b+=2;d=1/_nonzero(an*d+b);c=_nonzero(b+an/c);delta=d*c;h*=delta
            if abs(delta-1)<1e-14:return min(1.,max(0.,factor*h))
    raise ArithmeticError('Incomplete gamma did not converge')

def ranks(values):
    values=np.asarray(values);order=np.argsort(values,kind='stable');result=np.zeros(len(values));tie_total=0;start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]:end+=1
        result[order[start:end]]=(start+1+end)/2;t=end-start;tie_total+=t**3-t;start=end
    return result,tie_total

def friedman(matrix):
    x=np.asarray(matrix,dtype=float);n,k=x.shape
    if n<2 or k<3 or not np.isfinite(x).all():raise ValueError('Friedman requires at least two blocks and three models')
    ranked=[];ties=0
    for row in x:
        r,t=ranks(row);ranked.append(r);ties+=t
    totals=np.sum(ranked,axis=0);correction=1-ties/(n*k*(k*k-1))
    statistic=max(0.,(12*np.sum(totals**2)/(n*k*(k+1))-3*n*(k+1))/correction) if correction>0 else 0.
    return float(statistic),gamma_survival((k-1)/2,statistic/2)

def wilcoxon_exact(differences):
    d=np.asarray(differences,dtype=float);d=d[abs(d)>1e-14]
    if not len(d):return 1.
    if len(d)>20:raise ValueError('Exact signed-rank enumeration is limited to 20 independent blocks')
    r,_=ranks(abs(d));observed=abs(float(np.sum(np.sign(d)*r)))
    exceed=0;total=0
    for signs in product([-1,1],repeat=len(d)):
        exceed+=abs(float(np.asarray(signs)@r))>=observed-1e-12;total+=1
    return float(exceed/total)

def compare_panels(matrix,names):
    x=np.asarray(matrix,dtype=float);n,k=x.shape
    if n<2 or k<3 or len(names)!=k or not np.isfinite(x).all():raise ValueError('Invalid paired comparison panels')
    grand=x.mean();error=x-x.mean(0)-x.mean(1)[:,None]+grand
    effect=n*np.sum((x.mean(0)-grand)**2);residual=np.sum(error**2)
    f=float((effect/(k-1))/(residual/((n-1)*(k-1)))) if residual>1e-24 else None
    centered=x-x.mean(0);cov=centered.T@centered/(n-1);projection=np.eye(k)-np.ones((k,k))/k;s=projection@cov@projection
    denominator=(k-1)*np.trace(s@s);epsilon=float(np.clip(np.trace(s)**2/denominator,1/(k-1),1)) if denominator>0 else 1.
    fp=f_survival(f,epsilon*(k-1),epsilon*(k-1)*(n-1)) if f is not None else (0. if effect>1e-24 else 1.)
    statistic,p=friedman(x);pairs=[]
    for a,b in combinations(range(k),2):pairs.append({'a':names[a],'b':names[b],'p':wilcoxon_exact(x[:,a]-x[:,b])})
    previous=0.
    for rank,pair in enumerate(sorted(pairs,key=lambda row:row['p'])):
        previous=max(previous,min(1.,pair['p']*(len(pairs)-rank)));pair['holm_adjusted_p']=previous
    return {'models':names,'panels':n,'repeated_measures_ANOVA_F':f,'greenhouse_geisser_epsilon':epsilon,
            'ANOVA_GG_p':fp,'friedman_statistic':statistic,'friedman_p':p,
            'preferred_test':'Friedman; normal residuals are not assumed','posthoc_paired_wilcoxon_holm':pairs,
            'implementation':'manual ranks, tie correction, distribution tails, exact signed-rank enumeration'}
