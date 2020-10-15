<img src="..\images\image-20201015231458544.png" alt="image-20201015231458544" style="zoom: 80%;" />



# Policy Gradients  

## 1. Basic setting

*  **policy** is parametrized as $\pi_{\theta}(a|s)$ by some thing like neural nets.
* **Full likelihood:** $$\underbrace{p_{\theta}\left(\mathbf{s}_{1}, \mathbf{a}_{1}, \ldots, \mathbf{s}_{T}, \mathbf{a}_{T}\right)}_{p_{\theta}(\tau)}=p\left(\mathbf{s}_{1}\right) \prod_{t=1}^{T} \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right) p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)$$
* **Target:** find $$\theta^{\star}=\arg \max _{\theta} E_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right]$$,.      (where $\tau=(s_1,a_1,\cdots,s_T,a_T)$ and using notation $\tau$ means summation over time or product of probability)
* **Idea:** maximize expected cumulate reward $ E_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right]$ using <u>gradient methods</u>.

## 2. Direct policy differentiation 

#### 2.1 Reinforce algorithm

**Evaluate reward**: using MC method

 $J(\theta)=E_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right] \approx \frac{1}{N} \sum_{i} \sum_{t} r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)$, where $i$ for simulation and $t$ for time.

 

$$\begin{aligned} \nabla_{\theta} E_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right] & = \nabla_{\theta} J(\theta)\\&= \int \nabla_{\theta} p_{\theta}(\tau) r(\tau) d \tau\\&=\int p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) r(\tau) d \tau\\&=E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)\right]\end{aligned}$$

Notice $p_{\theta}(\tau) = p\left(\mathbf{s}_{1}\right) \prod_{t=1}^{T} \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right) p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)$, we have:

$$\nabla_{\theta} J(\theta)=E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)\right] = E_{\tau \sim p_{\theta}(\tau)}\left[\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right)\right]$$



To **estimate ** $\nabla_{\theta} J(\theta)$ , we still use MC method and average some simulation.

In conclusion:

---

$$\left\{\begin{array}{l}
\theta^{\star}=\arg \max _{\theta} J(\theta) \\
J(\theta)=E_{\tau \sim \pi_{\theta}(\tau)}[r(\tau)] \\
\nabla_{\theta} J(\theta)=E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)\right] =E_{\tau \sim p_{\theta}(\tau)}\left[\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right)\right]
\end{array}\right.$$

$$\begin{array}{l}
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N}\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)\right) \\
\theta \leftarrow \theta+\alpha \nabla_{\theta} J(\theta)
\end{array}$$

---

This is so-called **REINFORCE algorithm**:

<img src="..\images\image-20201015231608838.png" alt="image-20201015231608838" style="zoom:67%;" />



#### 2.2 Explanation:

* MLE: $\nabla_{\theta} J_{ML}(\theta)=E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau)\right] $

* Now: $\nabla_{\theta} J(\theta)=E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)\right] $

* > Making high reward trajectory more probable, low reward trajectory less probable.

  

#### 2.3 An example

Continuous policy (action is continuous)：A popular choice is Gaussian policies  using normal distribution.

<img src="..\images\image-20201015232042673.png" alt="image-20201015232042673" style="zoom:67%;" />

#### 2.4 Partial observability

$$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N}\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{o}_{i, t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)\right)$$

Works just fine.

#### 2.5 What is wrong?

<u>Above policy gradient may not work. What's wrong?</u>

For example, 

<img src="..\images\image-20201015234223736.png" alt="image-20201015234223736" style="zoom:67%;" />

Given three trajectory samples ($\tau_1,\tau_2,\tau_3$) with different reward $r(\tau_1),r(\tau_2),r(\tau_3)$ , one big negative and two small positive, represented by green bars:

Our new policy take smaller probability near $\tau_1$ and more near $\tau_2,\tau_3$.

<img src="..\images\image-20201015234627338.png" alt="image-20201015234627338" style="zoom: 50%;" />--new policy--》》<img src="..\images\image-20201015234635791.png" alt="image-20201015234635791" style="zoom:50%;" />



But if we set $r_i += C$ for some constant $C$ ( actually not change our problem ),

Our new policy take more near $\tau_1,\tau_2,\tau_3$ and become more spread out:

<img src="..\images\image-20201015234825900.png" alt="image-20201015234825900" style="zoom:50%;" /> --new policy--》》<img src="..\images\image-20201015234836875.png" alt="image-20201015234836875" style="zoom:50%;" />

Even worse: what if two "good" samples have $r(\tau_2)=r(\tau_3)=0$?: the behavior of new policy will be determined by $r(\tau_1)$ .

>  **Policy gradient has high variance!** 
>
> Different samples will lead to very different gradient.
>
> With limited samples size, the gradient can be really noisy and this method may not even converge.



## 3. Reduce variance

#### 3.1 Smaller variance with causality

<img src="..\images\image-20201016010743390.png" alt="image-20201016010743390" style="zoom:67%;" />

**Intuition:** In the new estimator, the sum of reward is reduced for every gradient. Smaller value may have smaller variance.

> Ref:  to be added

#### 3.2 Smaller variance with baseline

Recall that we have

$$\nabla_{\theta} J(\theta) = E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)\right]  \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} \log \pi_{\theta}(\tau_i)r(\tau_i)$$

The MC approximation is unbiased, but ( intuitively ) has high variance. How to improve?



Subtract the mean:

$$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} \log p_{\theta}(\tau_i)[r(\tau_i)-b],\ \ b\in\mathbb{R}$$

It is still **unbiased**:

$$ E\left[\nabla_{\theta} \log \pi_{\theta}(\tau) b\right]=\int \pi_{\theta}(\tau) \nabla_{\theta} \log \pi_{\theta}(\tau) b d \tau=\int \nabla_{\theta} \pi_{\theta}(\tau) b d \tau=b \nabla_{\theta} \int \pi_{\theta}(\tau) d \tau=b \nabla_{\theta} 1=0  $$



**Intuition:**

* If $b=E r(\tau)\approx\frac{1}{N}\sum_{i=1}^N r(\tau_i)$, then subtract average reward centralized our sample rewards.
* We can always reduce variance by subtracting a unbiased estimator of 0.



#### 3.1 Optimal baseline

Then we accuracy want to **minimize variance** by choosing $b$.

$$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} \log p_{\theta}(\tau_i)[r(\tau_i)-b],\ \ b\in\mathbb{R}$$

$$\text { Var }=E_{\tau \sim p_{\theta}(\tau)}\left[\left(\nabla_{\theta} \log p_{\theta}(\tau)(r(\tau)-b)\right)^{2}\right]-\left[E_{\tau \sim p_{\theta}(\tau)}\nabla_{\theta} \log p_{\theta}(\tau)(r(\tau)-b)\right]^{2}$$

The second term in $\text { Var }$ is independent from $b$ because $E_{\tau \sim p_{\theta}(\tau)}\nabla_{\theta} \log p_{\theta}(\tau)(r(\tau)-b) = E_{\tau \sim p_{\theta}(\tau)}\nabla_{\theta} \log p_{\theta}(\tau)r(\tau)$

Take derivate we have:

<img src="..\images\image-20201016005542396.png" alt="image-20201016005542396" style="zoom:67%;" />



In practice, you can just use the average reward.



#### 3.4 Conclusion

In practice, use average baseline ( alone do the most to make your policy gradient works nicely) and causality (easy and never hurts) , then your algorithm can be used to solve some meaningful RL problems.