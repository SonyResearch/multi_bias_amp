---
layout: post
title: "Men Also Do Laundry: Multi-Attribute Bias Amplification"
subtitle: "An interpretable metric for measuring bias amplification from multiple attributes"
date: 2023-06-30
published: true
authors: "Dora Zhao; Jerone Andrews; Alice Xiang"
author_links: "https://ai.sony/people/Dora-Zhao/;https://ai.sony/people/Jerone-Andrews/;https://ai.sony/people/Alice-Xiang/"
affiliations: "Sony AI, New York; Sony AI, Tokyo; Sony AI, New York"
code: "https://github.com/SonyResearch/multi_bias_amp"
paper: "https://arxiv.org/abs/2210.11924"
video: "https://icml.cc/virtual/2023/poster/25124"
dataset: ""
katex: True
---
<section>

    <p class="appendix-titles">BibTeX</p>

    <pre class="bibtex">

    @inproceedings{zhao2023men,
        title={Men Also Do Laundry: Multi-Attribute Bias Amplification},
        author={Zhao, Dora and Andrews, Jerone TA and Xiang, Alice},
        booktitle={International Conference on Machine Learning (ICML)},
        year={2023}
    }

</pre>

</section>

<section>
    <h2>Introduction</h2>
    <p>
        As computer vision systems become more widely deployed, there is increasing concern from both the research
        community and the public that these systems are not only reproducing but amplifying harmful social biases. The
        phenomenon of <em>bias amplification</em>, which is the focus of this work, refers to models amplifying inherent
        training set biases at test time{% cite zhao2017mals %}.
    </p>
    <p>{% katexmm %}Existing metrics{% cite zhao2017mals wang2021biasamp %} measure bias amplification with respect to
        single annotated attributes (e.g., $\texttt{computer}$). However, large-scale visual datasets often have
        multiple annotated attributes per image.<label for="coco-single" class="margin-toggle sidenote-number"></label>
        <input type="checkbox" id="coco-single" class="margin-toggle" /><span class="sidenote">For example, in COCO{%
            cite lin2014microsoft %} , 78.8% of the training set of images are associated with more than a single
            attribute (i.e., object).{% endkatexmm %}</span>
    </p>
    <p>
        {% katexmm %} 
        For example, in imSitu{% cite yatskar2016situation %}, individually the
        verb $\texttt{unloading}$ and location $\texttt{indoors}$ are skewed $\texttt{male}$. However, when considering
        $\{\texttt{unloading}, \texttt{indoors}\}$ in conjunction, the dataset is actually skewed $\texttt{female}$.
        Significantly, men tend to be pictured unloading <em>packages</em> outdoors whereas women are pictured unloading
        <em>laundry</em> or <em>dishes</em> indoors.
        {% endkatexmm %}
    </p>
    <figure>
        <img src="/assets/images/imSitu.png" />
        <figcaption>{% katexmm %}We provide bias scores (i.e., gender ratios) of the verbs $\texttt{pouring}$ and
            $\texttt{unloading}$ as well as location $\texttt{indoors}$ for $\texttt{female}$ gender expression in
            imSitu. While imSitu is skewed male for the single attributes, the multi-attributes (e.g.,
            $\{\texttt{pouring}, \texttt{indoors}\}$) are skewed $\texttt{female}$ ($\texttt{F}$). (Face pixelization is
            employed for privacy purposes.){% endkatexmm %}</figcaption>
    </figure>
</section>

<section>
    <h2>Key Takeaways</h2>
    <p>
        {% katexmm %} 
        1. Models can leverage correlations between a demographic group and multiple attributes simultaneously, suggesting that current <b>single-attribute metrics underreport the amount of bias being amplified</b>.
        {% endkatexmm %}
    </p>
    <p>
        {% katexmm %} 
        2. We validate our proposed metric through an analysis of gender bias amplification on the
        COCO{% cite lin2014microsoft %}, imSitu{% cite yatskar2016situation %}, and CelebA {% cite liu2015faceattributes %} datasets. On average, <b> bias amplification from multiple attributes is greater than that from single attributes</b>. 
        {% endkatexmm %}
    </p>
    <p>
        {% katexmm %}
        3. We benchmark proposed bias mitigation methods{% cite
        wang2019balanced zhao2017mals wang2020fair agarwal2022does %} using our metric and existing bias amplification
        metrics. Single-attribute <b>bias mitigation methods can inadvertently increase multi-attribute bias</b>.
        {% endkatexmm %}
    </p>
</section>

<section>
    <h2>Background</h2>
    <p>In their foundational work, Zhao et al.{% cite zhao2017mals %} propose a metric for bias amplification that
        measures the difference in object co-occurrences from the training to predicted distribution. Building on this
        work, Wang and Russakovsky{% cite wang2021biasamp %} proposed the metric <em>directional bias amplifcation</em>
        to disentangle bias arising from the attribute versus group prediction. An alternative line of work has focused
        on using <em>leakage</em>{% cite wang2019balanced hirota2022quantifying %}&#8212;the change in a classifier's
        ability to predict group membership from the training data to predictions. Our work extends
        <em>co-occurrence-based</em> metrics{% cite zhao2017mals wang2021biasamp %}.</p>
</section>

<section>
    <h2>Multi-Attribute Bias Amplification Metrics</h2>
    {% katexmm %}
    <p>
        We denote by $\mathcal{G}=\{g_1, \dots, g_t\}$ and ${\mathcal{A}=\{a_1,\dots,a_n\}}$ a set of $t$ group
        membership labels and a set of $n$ attributes, respectively. Here $g_i \in \mathcal{G}$ denotes group membership
        and ${a_i\in\mathcal{A}}$ denotes the absence (i.e., ${a_i=0}$) or presence (i.e., ${a_i=1}$) of attribute $i$
        in $x$.</p>

    <p>Let ${\mathcal{M}=\{m_1,\dots,m_{\ell}\}}$ denote a set of $\ell$ sets, containing all possible combinations of
        attributes, where $m_i$ is a set of attributes and $\lvert m_i \rvert \in \{1,\dots, n\}$.<label
            for="set-of-attrs" class="margin-toggle sidenote-number"></label><input type="checkbox" id="set-of-attrs"
            class="margin-toggle" /><span class="sidenote">For example, if $$\mathcal{A}=\{\texttt{bicycle},
            \texttt{car}, \texttt{motorcycle}\},$$ then$$
            \begin{aligned}
            \mathcal{M} = \{ & \{\texttt{bicycle}\}, \{\texttt{car}\}, \{\texttt{motorcycle}\},\\
            & \{\texttt{bicycle}, \texttt{car}\}, \{\texttt{bicycle}, \texttt{motorcycle}\}, \{\texttt{car},
            \texttt{motorcycle}\},\\
            & \{\texttt{bicycle}, \texttt{car}, \texttt{motorcycle}\} \}.
            \end{aligned}
            $$.</span> Note $m\in\mathcal{M}$ if and only if $\text{co-occur}(m, g) \geq 1$ in both the ground-truth
        training set and test set.
    </p>


    <p>
        We extend Zhao et al.'s{% cite zhao2017mals %} bias score to a multi-attribute setting such that the bias score
        of a set of attributes $m\in\mathcal{M}$ w.r.t. group $g\in\mathcal{G}$ is defined as:
        $$\text{bias}_{\text{train}}(m, g) = \dfrac{\text{co-occur}(m, g)}{\displaystyle\sum_{g^\prime \in \mathcal{G}}
        \text{co-occur}(m, g^\prime)},$$
        where $\text{co-occur}(m, g)$ denotes the number of times $m$ and $g$ co-occur in the training set.
    </p>
    {% endkatexmm %}

    <h3>Definition: Undirected Multi-Attribute Bias Amplification</h3>
    <p>
        {% katexmm %}
        We define our <em>undirected</em> multi-attribute bias amplification metric as:<label for="mals-ext"
            class="margin-toggle sidenote-number"></label>
        <input type="checkbox" id="mals-ext" class="margin-toggle" /><span class="sidenote">Our proposed metric extends
            Zhao et al.'s{% cite zhao2017mals %} single-attribute undirected bias amplification metric such that bias
            amplification from multiple attributes is measured.</span>
        $$\text{Multi}_\text{MALS} = X, \text{variance}(\Delta_{gm})$$
        where
        $$
        X = \frac{1}{\lvert \mathcal{M} \rvert}\sum_{g\in \mathcal{G} } \sum_{m\in \mathcal{M}}
        \left\lvert\Delta_{gm}\right\rvert
        $$
        and
        $$
        \begin{aligned}
            \Delta_{gm} &= \mathbb{1}\left[ \text{bias}_{\text{train}}(m, g) > \lvert
            \mathcal{G}\rvert^{-1} \right] \cdot \\ 
            &\left( \text{bias}_{\text{test}}(m, g) - \text{bias}_{\text{train}}(m, g)\right).
        \end{aligned}
        $$
        Here $\text{bias}_{\text{pred}}(m, g)$ denotes the bias score using the test set predictions of $m$ and $g$.
        $\text{Multi}_\text{MALS}$ measures both the mean and variance over the change in bias score from the training
        set ground truths to test set predictions.
        {% endkatexmm %}
    </p>




    <h3>Definition: Directional Multi-Attribute Bias Amplification</h3>
    <p>
        {% katexmm %}
        Let $\hat{m}$ and $\hat{g}$ denote a model's prediction for attribute group, $m$, and group membership, $g$,
        respectively.
        We define our <em>directional</em> multi-attribute bias amplification metric that takes into account the
        direction of bias as:<label for="mals-ext" class="margin-toggle sidenote-number"></label>
        <input type="checkbox" id="mals-ext" class="margin-toggle" /><span class="sidenote">Our proposed metric extends
            Wang and Russakovsky's{% cite wang2021biasamp %} single-attribute directional bias amplification metric such
            that bias amplification from multiple attributes is measured.</span>
        $$
        \text{Multi}_{\rightarrow} = X, \text{variance}(\Delta_{mg})
        $$
        where
        $$
        \begin{aligned}
         X &=\frac{1}{\lvert \mathcal{G} \rvert \lvert \mathcal{M} \rvert} \sum_{g\in \mathcal{G}}\sum_{m\in \mathcal{M}}  y_{gm}\left\lvert\Delta_{gm}\right\rvert+(1-y_{gm})\left\lvert-\Delta_{gm}\right\rvert,
        \end{aligned}
        $$
        $$
        \begin{aligned}
            y_{gm}=\mathbb{1} [ &P_{\text{train}}(g=1, m=1) \\ 
            &> P_{\text{train}}(g=1) P_{\text{train}}(m=1)],
        \end{aligned}
        $$
        and
        $$
        \begin{aligned}
            \Delta_{gm} &= \begin{cases}
            P_{\text{test}}(\hat{m}=1 \vert g=1) - P_{\text{train}}(m=1 \vert g=1) \\ \text{if measuring }G\rightarrow M\\
            P_{\text{test}}(\hat{g}=1 \vert m=1) - P_{\text{train}}(g=1 \vert m=1) \\
            \text{if measuring }M\rightarrow G
            \end{cases}
        \end{aligned}
        $$
        Unlike $\text{Multi}_\text{MALS}$, $\text{Multi}_{\rightarrow}$ captures both positive and negative
        correlations, i.e., $\text{Multi}_{\rightarrow}$ iterates over all $g\in\mathcal{G}$ regardless of whether
        $\text{bias}_{\text{train}}(m, g) > \lvert \mathcal{G}\rvert^{-1}$. Moreover, $\text{Multi}_{\rightarrow}$ takes
        into account the base rates for group membership and disentangles bias amplification arising from the group
        influencing the attribute(s) prediction ($\text{Multi}_{G \rightarrow M}$), as well as bias amplification from
        the attribute(s) influencing the group prediction ($\text{Multi}_{M \rightarrow G}$).
        {% endkatexmm %}
    </p>



</section>



<section>
    <h2>Comparison with Existing Metrics</h2>
    {% katexmm %}
    <p>Our proposed multi-attribute metric has the following advantages over existing single-attribute
        co-occurence-based metrics:</p>

    <p><b>Advantage 1: Our metric accounts for co-occurrences with multiple attributes.</b> 
        If a model, for example,
        learns the combination of $a_1$ and $a_2$, i.e., $m=\{a_1, a_2\}\in\mathcal{M}$, are correlated with
        $g\in\mathcal{G}$, it can exploit this correlation, potentially leading to bias amplification. By iterating over all $m \in \mathcal{M}$, our proposed metric accounts for amplification from
        single and multiple attributes. Thus, capturing sets of attributes exhibiting amplification, which are not
        accounted for by existing metrics.
    </p>
    {% endkatexmm %}

    <p><b>Advantage 2: Negative and positive values do not cancel each other out.</b>
        {% katexmm %}
        Existing metrics calculate bias amplification by aggregating over the difference in bias scores for each
        attribute individually. Suppose there is a dataset with two annotated attributes $a_1$ and $a_2$. It is possible
        that ${\Delta_{ga_1} \approx -\Delta_{ga_2}}$. Since our metrics use absolute values, we ensure that positive and negative bias amplifications per attribute do
        not cancel each other out. </p>
    {% endkatexmm %}

    <p><b>Advantage 3: Our metric is more interpretable.</b>
        {% katexmm %}
        There is a lack of intuition as to what an &ldquo;ideal&rdquo; amplification value is. One interpretation is
        that smaller values are more desirable. This becomes less clear when values are negative. First, there often
        exists a trade-off between performance and smaller bias amplification values. Second, high magnitude negative
        bias amplification may lead to erasure of certain groups.<label for="imsitu-type-f"
            class="margin-toggle sidenote-number"></label>
        <input type="checkbox" id="imsitu-type-f" class="margin-toggle" /><span class="sidenote">For example, in imSitu,
            the $\text{bias}_{\text{train}}(\{\texttt{typing}\}, \texttt{F}) = \textsf{0.52}$. Negative bias
            amplification signifies the model underpredicts $(\{\texttt{typing}\}, \texttt{F})$, which could reinforce
            negative gender stereotypes{% cite zhao2017mals %}.</span> Instead, we may want to minimize the distance
        between the bias amplification value and zero. This interpretation offers the advantage that large negative
        values are also not desirable. However, a potential dilemma occurs when interpreting two values with the same
        magnitude but opposite signs, which is a value-laden decision and depends on the system's context.
        {% endkatexmm %}
    </p>


    <p>Our proposed metric is easy to interpret. Since we use absolute differences, the ideal value is unambiguously
        zero. Further, reporting variance provides intuition about whether amplification is uniform across all
        attributes or if particular attributes are more amplified.
    </p>
</section>

<section>
    <p class="appendix-titles">References</p>

    {% bibliography --cited_in_order %}

</section>