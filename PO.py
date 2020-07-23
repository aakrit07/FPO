#!/usr/bin/env python3
import pandas as pd
import numpy as np
import scipy.optimize as sco

np.set_printoptions(precision = 6 , suppress = True)

df = pd.read_csv("C:\\Users\\aakri\\source\\repos\\Portfolio\\Data\\NIFTY_50.csv")
df = df.drop('Date', 1)

df_cluster1 = pd.read_csv("C:\\Users\\aakri\\source\\repos\\Portfolio\\Data\\df_cluster1.csv")
df_cluster3 = pd.read_csv("C:\\Users\\aakri\\source\\repos\\Portfolio\\Data\\df_cluster3.csv")
df_cluster4 = pd.read_csv("C:\\Users\\aakri\\source\\repos\\Portfolio\\Data\\df_cluster4.csv")
df_cluster6 = pd.read_csv("C:\\Users\\aakri\\source\\repos\\Portfolio\\Data\\df_cluster6.csv")

df_cluster1 = df_cluster1.drop('Date', 1)
df_cluster3 = df_cluster3.drop('Date', 1)
df_cluster4 = df_cluster4.drop('Date', 1)
df_cluster6 = df_cluster6.drop('Date', 1)

clusters = [df_cluster1,df_cluster3,df_cluster4,df_cluster6]

def log_Returns(clusters):
    log_returns = []
    for i in clusters:
        log_ret = np.log(i/i.shift(1))
        log_ret_mean=log_ret.mean()
        log_returns.append(log_ret)
    return log_returns

log_Returns = log_Returns(clusters)

def cov_Matrix(log_Returns):
    cov_matrix = []
    for i in log_Returns:
        cov_mat = i.cov()
        cov_mat.style.background_gradient(cmap='coolwarm')
        cov_matrix.append(cov_mat)
    return cov_matrix

cov_Matrix = cov_Matrix(log_Returns)

def stock_Returns(clusters):
    stock_returns = []
    for i in clusters:
        stock_ret = i.pct_change()
        stock_ret = stock_ret.round(4)*100
        stock_returns.append(stock_ret)
    return stock_returns

stock_Returns = stock_Returns(clusters)

df_latest = df.drop(["CIPLA.NS", "COALINDIA.NS","ONGC.NS","SUNPHARMA.NS","TATAMOTORS.NS"], axis=1)
Stock_Price_Latest = list(df_latest.iloc[-1,:])

def Amount_per_Cluster(Investment,clusters):
    Amount_per_Cluster = Investment / clusters
    Amount_per_Cluster = round(Amount_per_Cluster,2)
    return Amount_per_Cluster


def Investment(Max_Sharpe_Opt):
    Investment = []
    for i in Max_Sharpe_Opt:
        Investment.append(i * Amount)
    return Investment

No_of_Clusters = len(clusters)
cluster_length = []        
for j in range(No_of_Clusters):
    cluster_len = len(clusters[j].columns)
    cluster_length.append(cluster_len)

    
def weights_Clusters(cluster_length):
    weights = []
    for k in cluster_length:
        weightage = np.array(np.random.dirichlet(np.ones(k), size=1))
        weights.append(weightage)
    return weights

weights = weights_Clusters(cluster_length)

# ## OPTIMIZATION

#a) Portfolio with Maximum Sharpe Ratio using Optimization
def Max_Sharpe_Portfolio(Invest,clusters):
    

    No_of_Clusters = len(clusters)
    Amount = Amount_per_Cluster(Invest,No_of_Clusters)
    port = []
    cluster_length = []
    columns_portfolio = []
    Max_Sharpe_Opt = []
    Investment_Assets = []
    cluster_out = []
    cluster_volatility = []
    cluster_returns = []
    cluster_sharpe = []

    def Investment(Max_Sharpe_Opt):
        Investment = []
        for i in Max_Sharpe_Opt:
            Investment.append(i * Amount)
        return Investment

    Investment_df = pd.DataFrame(columns = ["Assets","Weights","Amount Invested"])
    Inv_df = pd.DataFrame(columns = ["Weights","Amount Invested"])


    for i in clusters:
        for j in range(No_of_Clusters):
            cluster_len = len(clusters[j].columns)
            cluster_length.append(cluster_len)
            stock_returns = stock_Returns[j]
            cov_matrix = cov_Matrix[j]
            log_returns = log_Returns[j]
            weights_cluster = weights[j]  
            k = cluster_length[j]

            def optim_info(weights_cluster, stock_returns, cov_matrix):
                pvol = np.sqrt(np.dot(weights_cluster.T,np.dot(cov_matrix, weights_cluster)))
                pret = np.sum(weights_cluster*log_returns.mean())*252
                return np.array([pret,pvol,(pret)/pvol])

            def neg_sharpe_ratio(weights_cluster, stock_returns):
                return -optim_info(weights_cluster,stock_returns, cov_matrix)[2]
                
            args = (log_returns.mean(), cov_matrix)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            bound = (0.0,1.0)
            bounds = tuple(bound for x in range(k))
            optimized = sco.minimize(neg_sharpe_ratio, weights_cluster, args=log_returns,method='SLSQP', bounds=bounds, constraints=constraints)

            Max_Sharpe_Opt_Weights = pd.DataFrame(optimized['x'].round(2),stock_returns.columns,columns=['Max_Sharpe_Opt_Weights'])
            Max_Sharpe_Opt_clusters = Max_Sharpe_Opt_Weights["Max_Sharpe_Opt_Weights"]
            Max_Sharpe_Opt.append(Max_Sharpe_Opt_clusters)
            Investment_clusters = Investment(Max_Sharpe_Opt_clusters)
            Investment_Assets.append(Investment_clusters)
            cluster_out.append(optim_info(optimized['x'], log_returns, cov_matrix))
            

    Max_Sharpe_Opt = Max_Sharpe_Opt[0:No_of_Clusters]
    Investment_Assets = Investment_Assets[0:No_of_Clusters]

    Inv_df["Weights"] = Max_Sharpe_Opt
    Inv_df["Amount Invested"] = Investment_Assets

    for k in range(No_of_Clusters):
        wts= Inv_df.loc[Inv_df.index[k], 'Weights']
        dfw_dict=wts.to_dict()
        df_res = pd.DataFrame(list(dfw_dict.items()),columns = ['Assets','Weights']) 
        inv= Inv_df.loc[Inv_df.index[k], 'Amount Invested']
        df_inv=pd.DataFrame (inv,columns=['Amount Invested'])
        Cluster_Inv_df = pd.concat([df_res,df_inv], axis=1, sort=False)
        Investment_df = Investment_df.append(Cluster_Inv_df)

    Investment_df = Investment_df.sort_values(by=["Assets"])
    Investment_df["Stock Price"] = Stock_Price_Latest
    Investment_df["Number of Shares per Asset"] = round(Investment_df["Amount Invested"]/Investment_df["Stock Price"],0)
    Investment_df["Amount Left"] = Investment_df["Amount Invested"] - (Investment_df["Stock Price"] * Investment_df["Number of Shares per Asset"])
    Investment_df = Investment_df[Investment_df['Weights'] != 0]
    Investment_df.reset_index(drop=True, inplace=True)
    cluster_out = cluster_out[0:No_of_Clusters]
    for l in range(No_of_Clusters):
        cluster_returns.append(cluster_out[l][0])
        cluster_volatility.append(cluster_out[l][1])
        cluster_sharpe.append(cluster_out[l][2])
        
    global Portfolio_Returns_MS,Portfolio_Volatility_MS,Portfolio_Sharpe_MS

    Portfolio_Returns_MS = (sum(cluster_returns)/No_of_Clusters) * 100
    Portfolio_Volatility_MS = (sum(cluster_volatility)/No_of_Clusters) * 100
    Portfolio_Sharpe_MS = (sum(cluster_sharpe)/No_of_Clusters)

    Amount_Left = round(Investment_df["Amount Left"].sum(),2)
    Total_Amount = Invest - Amount_Left
    
    amt= "Total Amount Invested : " +str(Total_Amount)
    rtn ="The Expected Portfolio Return is : " +str(Portfolio_Returns_MS)
    risk = "The Expected Portfolio Risk is : " +str(Portfolio_Volatility_MS)
    max = "The Expected Portfolio Sharpe Ratio is : " +str(Portfolio_Sharpe_MS)
    
    msg = [amt,rtn,risk,max,Investment_df]
    return msg

#b)  Portfolio with Minimum Volatility using Optimization
def Min_Volatility_Portfolio(Invest,clusters):
    

    No_of_Clusters = len(clusters)
    Amount = Amount_per_Cluster(Invest,No_of_Clusters)
    port = []
    cluster_length = []
    columns_portfolio = []
    Min_Vol_Opt = []
    Investment_Assets = []
    cluster_out = []
    cluster_volatility = []
    cluster_returns = []
    cluster_sharpe = []

    def Investment(Min_Vol_Opt):
        Investment = []
        for i in Min_Vol_Opt:
            Investment.append(i * Amount)
        return Investment

    Investment_df = pd.DataFrame(columns = ["Assets","Weights","Amount Invested"])
    Inv_df = pd.DataFrame(columns = ["Weights","Amount Invested"])


    for i in clusters:
        for j in range(No_of_Clusters):
            cluster_len = len(clusters[j].columns)
            cluster_length.append(cluster_len)
            stock_returns = stock_Returns[j]
            cov_matrix = cov_Matrix[j]
            log_returns = log_Returns[j]
            weights_cluster = weights[j]  
            k = cluster_length[j]
            
            def optim_info(weights_cluster, stock_returns, cov_matrix):
                pvol = np.sqrt(np.dot(weights_cluster.T,np.dot(cov_matrix, weights_cluster)))
                pret = np.sum(weights_cluster*log_returns.mean())*252
                return np.array([pret,pvol,(pret)/pvol])
            
            def portfolio_volatility(weights_cluster, mean_returns, cov_matrix):
                return optim_info(weights_cluster, mean_returns, cov_matrix)[1]


            args = (log_returns.mean(), cov_matrix)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = (0.0,1.0)
            bounds = tuple(bound for x in range(k))
            optimized1 = sco.minimize(portfolio_volatility, k*[1./k,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

            Min_Vol_Opt_Weights = pd.DataFrame(optimized1['x'].round(2),stock_returns.columns, columns=['Min_Vol_Opt_Weights'])
            Min_Vol_Opt_Weights_clusters = Min_Vol_Opt_Weights["Min_Vol_Opt_Weights"]
            Min_Vol_Opt.append(Min_Vol_Opt_Weights_clusters)
            Investment_clusters = Investment(Min_Vol_Opt_Weights_clusters)
            Investment_Assets.append(Investment_clusters)
            cluster_out.append(optim_info(optimized1['x'], log_returns, cov_matrix))
            
            
    Min_Vol_Opt = Min_Vol_Opt[0:No_of_Clusters]
    Investment_Assets = Investment_Assets[0:No_of_Clusters]


    Inv_df["Weights"] = Min_Vol_Opt
    Inv_df["Amount Invested"] = Investment_Assets

    for k in range(No_of_Clusters):
        wts= Inv_df.loc[Inv_df.index[k], 'Weights']
        dfw_dict=wts.to_dict()
        df_res = pd.DataFrame(list(dfw_dict.items()),columns = ['Assets','Weights']) 
        inv= Inv_df.loc[Inv_df.index[k], 'Amount Invested']
        df_inv=pd.DataFrame (inv,columns=['Amount Invested'])
        Cluster_Inv_df = pd.concat([df_res,df_inv], axis=1, sort=False)
        Investment_df = Investment_df.append(Cluster_Inv_df)

    Investment_df = Investment_df.sort_values(by=["Assets"])
    Investment_df["Stock Price"] = Stock_Price_Latest
    Investment_df["Number of Shares per Asset"] = round(Investment_df["Amount Invested"]/Investment_df["Stock Price"],0)
    Investment_df["Amount Left"] = Investment_df["Amount Invested"] - (Investment_df["Stock Price"] * Investment_df["Number of Shares per Asset"])
    Investment_df = Investment_df[Investment_df['Weights'] != 0]
    Investment_df.reset_index(drop=True, inplace=True)
    cluster_out = cluster_out[0:No_of_Clusters]
    for l in range(No_of_Clusters):
        cluster_returns.append(cluster_out[l][0])
        cluster_volatility.append(cluster_out[l][1])
        cluster_sharpe.append(cluster_out[l][2])
    
    global Portfolio_Returns_MV,Portfolio_Volatility_MV,Portfolio_Sharpe_MV
    
    Portfolio_Returns_MV = (sum(cluster_returns)/No_of_Clusters) * 100
    Portfolio_Volatility_MV = (sum(cluster_volatility)/No_of_Clusters) * 100
    Portfolio_Sharpe_MV = (sum(cluster_sharpe)/No_of_Clusters)

    Amount_Left = round(Investment_df["Amount Left"].sum(),2)
    Total_Amount = Invest - Amount_Left
    
    amt= "Total Amount Invested : " +str(Total_Amount)
    rtn ="The Expected Portfolio Return is : " +str(Portfolio_Returns_MV)
    risk = "The Expected Portfolio Risk is : " +str(Portfolio_Volatility_MV)
    max = "The Expected Portfolio Sharpe Ratio is : " +str(Portfolio_Sharpe_MV)
    
    msg = [amt,rtn,risk,max,Investment_df]
    
    return msg

#c) Portfolio with Maximum Returns using Optimization
def Max_Returns_Portfolio(Invest,clusters):
    

    No_of_Clusters = len(clusters)
    Amount = Amount_per_Cluster(Invest,No_of_Clusters)
    port = []
    cluster_length = []
    columns_portfolio = []
    Max_Ret_Opt = []
    Investment_Assets = []
    cluster_out = []
    cluster_volatility = []
    cluster_returns = []
    cluster_sharpe = []

    def Investment(Max_Ret_Opt):
        Investment = []
        for i in Max_Ret_Opt:
            Investment.append(i * Amount)
        return Investment

    Investment_df = pd.DataFrame(columns = ["Assets","Weights","Amount Invested"])
    Inv_df = pd.DataFrame(columns = ["Weights","Amount Invested"])


    for i in clusters:
        for j in range(No_of_Clusters):
            cluster_len = len(clusters[j].columns)
            cluster_length.append(cluster_len)
            stock_returns = stock_Returns[j]
            cov_matrix = cov_Matrix[j]
            log_returns = log_Returns[j]
            weights_cluster = weights[j]  
            k = cluster_length[j]
            
            def optim_info(weights_cluster, stock_returns, cov_matrix):
                pvol = np.sqrt(np.dot(weights_cluster.T,np.dot(cov_matrix, weights_cluster)))
                pret = np.sum(weights_cluster*log_returns.mean())*252
                return np.array([pret,pvol,(pret)/pvol])
            
            def max_returns(weights_cluster, mean_returns, cov_matrix):
                return - optim_info(weights_cluster, mean_returns, cov_matrix)[0]


            args = (log_returns.mean(), cov_matrix)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = (0.0,1.0)
            bounds = tuple(bound for x in range(k))
            optimized2 = sco.minimize(max_returns, k*[1./k,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

            Max_Ret_Opt_Weights = pd.DataFrame(optimized2['x'].round(2),stock_returns.columns, columns=['Max_Ret_Opt_Weights'])
            Max_Ret_Opt_Weights_clusters = Max_Ret_Opt_Weights["Max_Ret_Opt_Weights"]
            Max_Ret_Opt.append(Max_Ret_Opt_Weights_clusters)
            Investment_clusters = Investment(Max_Ret_Opt_Weights_clusters)
            Investment_Assets.append(Investment_clusters)
            cluster_out.append(optim_info(optimized2['x'], log_returns, cov_matrix))
            
            
    Max_Ret_Opt = Max_Ret_Opt[0:No_of_Clusters]
    Investment_Assets = Investment_Assets[0:No_of_Clusters]


    Inv_df["Weights"] = Max_Ret_Opt
    Inv_df["Amount Invested"] = Investment_Assets

    for k in range(No_of_Clusters):
        wts= Inv_df.loc[Inv_df.index[k], 'Weights']
        dfw_dict=wts.to_dict()
        df_res = pd.DataFrame(list(dfw_dict.items()),columns = ['Assets','Weights']) 
        inv= Inv_df.loc[Inv_df.index[k], 'Amount Invested']
        df_inv=pd.DataFrame (inv,columns=['Amount Invested'])
        Cluster_Inv_df = pd.concat([df_res,df_inv], axis=1, sort=False)
        Investment_df = Investment_df.append(Cluster_Inv_df)

    Investment_df = Investment_df.sort_values(by=["Assets"])
    Investment_df["Stock Price"] = Stock_Price_Latest
    Investment_df["Number of Shares per Asset"] = round(Investment_df["Amount Invested"]/Investment_df["Stock Price"],0)
    Investment_df["Amount Left"] = Investment_df["Amount Invested"] - (Investment_df["Stock Price"] * Investment_df["Number of Shares per Asset"])
    Investment_df = Investment_df[Investment_df['Weights'] != 0]
    Investment_df.reset_index(drop=True, inplace=True)
    cluster_out = cluster_out[0:No_of_Clusters]
    for l in range(No_of_Clusters):
        cluster_returns.append(cluster_out[l][0])
        cluster_volatility.append(cluster_out[l][1])
        cluster_sharpe.append(cluster_out[l][2])
    
    global Portfolio_Returns_MR,Portfolio_Volatility_MR,Portfolio_Sharpe_MR
    Portfolio_Returns_MR = (sum(cluster_returns)/No_of_Clusters) * 100
    Portfolio_Volatility_MR = (sum(cluster_volatility)/No_of_Clusters) * 100
    Portfolio_Sharpe_MR = (sum(cluster_sharpe)/No_of_Clusters)

    Amount_Left = round(Investment_df["Amount Left"].sum(),2)
    Total_Amount = Invest - Amount_Left
        
    amt= "Total Amount Invested : " +str(Total_Amount)
    rtn ="The Expected Portfolio Return is : " +str(Portfolio_Returns_MR)
    risk = "The Expected Portfolio Risk is : " +str(Portfolio_Volatility_MR)
    max = "The Expected Portfolio Sharpe Ratio is : " +str(Portfolio_Sharpe_MR)
    
    msg = [amt,rtn,risk,max,Investment_df]
    
    return msg
    

#d) Portfolio with Investor Specified Return Level using Optimization

def Investor_Specified_Returns_Portfolio(Invest,clusters,target):
    

    No_of_Clusters = len(clusters)
    Amount = Amount_per_Cluster(Invest,No_of_Clusters)
    port = []
    cluster_length = []
    columns_portfolio = []
    Eff_Port_Weights = []
    Investment_Assets = []
    cluster_out = []
    cluster_volatility = []
    cluster_returns = []
    cluster_sharpe = []

    def Investment(Eff_Port_Weights):
        Investment = []
        for i in Eff_Port_Weights:
            Investment.append(i * Amount)
        return Investment

    Investment_df = pd.DataFrame(columns = ["Assets","Weights","Amount Invested"])
    Inv_df = pd.DataFrame(columns = ["Weights","Amount Invested"])


    for i in clusters:
        for j in range(No_of_Clusters):
            cluster_len = len(clusters[j].columns)
            cluster_length.append(cluster_len)
            stock_returns = stock_Returns[j]
            cov_matrix = cov_Matrix[j]
            log_returns = log_Returns[j]
            weights_cluster = weights[j]  
            k = cluster_length[j]
            
            def optim_info(weights_cluster, stock_returns, cov_matrix):
                pvol = np.sqrt(np.dot(weights_cluster.T,np.dot(cov_matrix, weights_cluster)))
                pret = np.sum(weights_cluster*log_returns.mean())*252
                return np.array([pret,pvol,(pret)/pvol])
            
            def portfolio_volatility(weights_cluster, mean_returns, cov_matrix):
                return optim_info(weights_cluster, mean_returns, cov_matrix)[1]
            
            def efficient_return(log_returns, cov_matrix, target):
                num_assets = len(log_returns)
                args = (log_returns, cov_matrix)

                def portfolio_return(weights_cluster):
                    return optim_info(weights_cluster, log_returns, cov_matrix)[0]

                constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                           {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0,1) for asset in range(k))
                eff_port_return = sco.minimize(portfolio_volatility, weights_cluster, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                return eff_port_return

            Target_Portfolio = efficient_return(log_returns.mean(),cov_matrix,target) #Adjust Last Param for Target Vol
            Efficient_Portfolio_Weights = pd.DataFrame(Target_Portfolio['x'].round(3),stock_returns.columns, columns=['Efficient_Portfolio_Weights'])
            Eff_Port_Weights_clusters = Efficient_Portfolio_Weights["Efficient_Portfolio_Weights"]
            Eff_Port_Weights.append(Eff_Port_Weights_clusters)
            Investment_clusters = Investment(Eff_Port_Weights_clusters)
            Investment_Assets.append(Investment_clusters)
            cluster_out.append(optim_info(Target_Portfolio['x'], log_returns, cov_matrix))


            
            
    Eff_Port_Weights = Eff_Port_Weights[0:No_of_Clusters]
    Investment_Assets = Investment_Assets[0:No_of_Clusters]


    Inv_df["Weights"] = Eff_Port_Weights
    Inv_df["Amount Invested"] = Investment_Assets

    for k in range(No_of_Clusters):
        wts= Inv_df.loc[Inv_df.index[k], 'Weights']
        dfw_dict=wts.to_dict()
        df_res = pd.DataFrame(list(dfw_dict.items()),columns = ['Assets','Weights']) 
        inv= Inv_df.loc[Inv_df.index[k], 'Amount Invested']
        df_inv=pd.DataFrame (inv,columns=['Amount Invested'])
        Cluster_Inv_df = pd.concat([df_res,df_inv], axis=1, sort=False)
        Investment_df = Investment_df.append(Cluster_Inv_df)

    Investment_df = Investment_df.sort_values(by=["Assets"])
    Investment_df["Stock Price"] = Stock_Price_Latest
    Investment_df["Number of Shares per Asset"] = round(Investment_df["Amount Invested"]/Investment_df["Stock Price"],0)
    Investment_df["Amount Left"] = Investment_df["Amount Invested"] - (Investment_df["Stock Price"] * Investment_df["Number of Shares per Asset"])
    Investment_df = Investment_df[Investment_df['Weights'] != 0]
    Investment_df.reset_index(drop=True, inplace=True)
    cluster_out = cluster_out[0:6]
    for l in range(No_of_Clusters):
        cluster_returns.append(cluster_out[l][0])
        cluster_volatility.append(cluster_out[l][1])
        cluster_sharpe.append(cluster_out[l][2])
    
    global Portfolio_Returns_ISRP,Portfolio_Volatility_ISRP,Portfolio_Sharpe_ISRP
    Portfolio_Returns_ISRP = (sum(cluster_returns)/No_of_Clusters) * 100
    Portfolio_Volatility_ISRP = (sum(cluster_volatility)/No_of_Clusters) * 100
    Portfolio_Sharpe_ISRP = (sum(cluster_sharpe)/No_of_Clusters)

    Amount_Left = round(Investment_df["Amount Left"].sum(),2)
    Total_Amount = Invest - Amount_Left
    
    amt= "Total Amount Invested : " +str(Total_Amount)
    rtn ="The Expected Portfolio Return is : " +str(Portfolio_Returns_ISRP)
    risk = "The Expected Portfolio Risk is : " +str(Portfolio_Volatility_ISRP)
    max = "The Expected Portfolio Sharpe Ratio is : " +str(Portfolio_Sharpe_ISRP)
    
    msg = [amt,rtn,risk,max,Investment_df]
    
    return msg
