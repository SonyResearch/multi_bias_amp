---
layout: post
title: "Men Also Do Laundry: Multi-Attribute Bias Amplification"
subtitle: "Interpretable metrics for measuring bias amplification from multiple attributes"
conference: ICML 2023
published: true
authors: "Dora Zhao; Jerone Andrews; Alice Xiang"
author_links: "https://ai.sony/people/Dora-Zhao/;https://ai.sony/people/Jerone-Andrews/;https://ai.sony/people/Alice-Xiang/"
affiliations: "Sony AI;Sony AI;Sony AI"
code: "https://github.com/SonyResearch/multi_bias_amp"
paper: "https://arxiv.org/abs/2210.11924"
poster: ""
video: "https://icml.cc/virtual/2023/poster/25124"
dataset: ""
bibtex: "{{ 'assets/bib/multi_bias_amp.bib' | prepend: site.github_url }}"
acknowledgments: This work was funded by Sony Research Inc. We thank William Thong and Julienne LaChance for their helpful comments and suggestions.
katex: True
---


<div class="inner">
    <h2 id="h2-first">Background and related work</h2>
    <p>
        As computer vision systems become more widely deployed, there is increasing concern from both the research
        community and the public that these systems are not only reproducing but <i>amplifying</i> harmful social biases. The
        phenomenon of <i>bias amplification</i>, which is the focus of our work, refers to models amplifying inherent
        training set biases at test time{% cite zhao2017mals %}.
    </p>
    <p>
    There are two main approaches to quantifying bias amplification in computer vision models:</p>
    <ul>
        <li><i>Leakage-based metrics</i>{% cite wang2019balanced hirota2022quantifying %}, measuring the change in a classifier’s ability to predict group membership from the training data to predictions.</li>
        <li><i>Co-occurrence-based metrics</i>{% cite zhao2017mals wang2021biasamp %}, measuring the change in ratio of a group and single attribute from the training data to predictions.</li>
    </ul>
    <p>
        {% katexmm %}
        Existing metrics{% cite zhao2017mals wang2021biasamp %} measure bias amplification with respect to
        single annotated attributes (e.g., `computer`). However, large-scale visual datasets often have
        multiple annotated attributes per image. For example, in the COCO dataset{% cite lin2014microsoft %}, 78.8&#37; of the training set of images are associated with more than a single attribute (i.e., object).
        {% endkatexmm %}
    </p>
    <p>
        {% katexmm %} 
        More importantly, considering multiple attributes can reveal additional nuances not present when considering only single attributes. In the imSitu dataset{% cite yatskar2016situation %}, individually the verb `unloading` and location `indoors` are skewed `male`. However, when considering {`unloading`, `indoors`} in conjunction, the dataset is actually skewed `female`.
        Significantly, men tend to be pictured unloading <i>packages</i> `outdoors` whereas women are pictured `unloading`
        <i>laundry</i> or <i>dishes</i> `indoors`. Even when men are pictured `indoors`, they are `unloading` <i>boxes</i> or <i>equipment</i> as opposed to laundry or dishes. Models can similarly leverage correlations between a group and either single or multiple attributes simultaneously.
        {% endkatexmm %}
    </p>
    <figure class="mb0">
        <img src="{{ 'assets/images/imSitu.png' | prepend: site.github_url }}"/>
        <figcaption>
            Bias scores (i.e., gender ratios) of the verbs `pouring` and `unloading` as well as location `indoors` in imSitu. While imSitu is skewed `male` for the single attributes, the multi-attributes (e.g., {`pouring`, `indoors`}) are skewed `female`. Note that we have replaced imSitu images with <a href="https://stock.adobe.com/" target="_blank">Adobe Stock</a> images for privacy reasons.
            </figcaption>
    </figure>
    <h2>Key takeaways</h2>
    <p>We propose two multi-attribute bias amplification metrics that evaluate bias arising from both single and multiple attributes.</p>
    <ul>
        <li>We are the first to study multi-attribute bias amplification, highlighting that models can leverage correlations between a demographic group and multiple attributes simultaneously, suggesting that current <i>single-attribute metrics underreport the amount of bias being amplified</i>.</li>
        <li>When comparing the performance of multi-label classifiers trained on the COCO, imSitu, and CelebA{% cite liu2015faceattributes %} datasets, we empirically demonstrate that, on average, <i>bias amplification from multiple attributes is greater than that from single attributes</i>.</li>
        <li>While prior works have demonstrated that models can learn to exploit different spurious correlations if one is mitigated{%cite li2022whac li2022discover %}, we are the first to demonstrate that <i>bias mitigation methods</i>{% cite zhao2017mals wang2019balanced wang2020fair agarwal2022does %} <i>for single attribute bias can inadvertently increase multi-attribute bias</i>.</li>
    </ul>
    <h2>Multi-attribute bias amplification metrics</h2>
    {% katexmm %}
    <p>
        We denote by $\mathcal{G}=\{g_1, \dots, g_t\}$ and ${\mathcal{A}=\{a_1,\dots,a_n\}}$ a set of $t$ group membership labels and a set of $n$ attributes, respectively. Let ${\mathcal{M}=\{m_1,\dots,m_{\ell}\}}$ denote a set of $\ell$ sets, containing all possible combinations of attributes, where $m_i$ is a set of attributes and $\lvert m_i \rvert \in \{1,\dots, n\}$. Note $m\in\mathcal{M}$ if and only if $\text{co-occur}(m, g) \geq 1$ in both the ground-truth training set and test set.
    </p>
    <p>
        We extend Zhao et al.'s{% cite zhao2017mals %} bias score to a multi-attribute setting such that the bias score
        of a set of attributes $m\in\mathcal{M}$ with respect to group $g\in\mathcal{G}$ is defined as:
        $$\text{bias}_{\text{train}}(m, g) = \dfrac{\text{co-occur}(m, g)}{\displaystyle\sum_{g^\prime \in \mathcal{G}}
        \text{co-occur}(m, g^\prime)},$$
        where $\text{co-occur}(m, g)$ denotes the number of times $m$ and $g$ co-occur in the training set.
    </p>
    {% endkatexmm %}
    <p class="mt80"><b>Undirected multi-attribute bias amplification metric</b></p>
    <p class="mt20">
    <p>
        {% katexmm %}
        Our proposed <i>undirected</i> multi-attribute bias amplification metric extends
            Zhao et al.'s{% cite zhao2017mals %} single-attribute metric such that bias
            amplification from multiple attributes is measured:
        $$\text{Multi}_\text{MALS} = X, \text{variance}(\Delta_{gm})$$
        where
        $$
        X = \frac{1}{\lvert \mathcal{M} \rvert}\sum_{g\in \mathcal{G} } \sum_{m\in \mathcal{M}}
        \left\lvert\Delta_{gm}\right\rvert
        $$
        and
        $$
        \begin{aligned}
            \Delta_{gm} &= \mathbf{1}\left[ \text{bias}_{\text{train}}(m, g) > \lvert
            \mathcal{G}\rvert^{-1} \right] \cdot \\ &\left( \text{bias}_{\text{test}}(m, g) - \text{bias}_{\text{train}}(m, g)\right).
        \end{aligned}
        $$
        Here $\mathbf{1} [\cdot]$ and $\text{bias}_{\text{pred}}(m, g)$ denote an indicator function and the bias score using the test set predictions of $m$ and $g$, respectively.</p>
        <p>$\text{Multi}_\text{MALS}$ measures both the mean and variance over the change in bias score from the training
        set ground truths to test set predictions. By definition, $\text{Multi}_\text{MALS}$ only captures group membership labels that are positively correlated with a set of attributes, i.e., due to the constraint that $\text{bias}_{\text{train}}(m,g) > \lvert \mathcal{G}\rvert^{-1}.$</p>
        {% endkatexmm %}
    </p>
    <p class="mt80"><b>Directional multi-attribute bias amplification metric</b></p>
    <p class="mt20">
        {% katexmm %}
        Let $\hat{m}$ and $\hat{g}$ denote a model's prediction for attribute group, $m$, and group membership, $g$,
        respectively. Our proposed <i>directional</i> multi-attribute bias amplification metric extends Wang and Russakovsky's{% cite wang2021biasamp %} single-attribute metric such that bias amplification from multiple attributes is measured:
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
            y_{gm}=\mathbf{1} [ &P_{\text{train}}(g=1, m=1) > P_{\text{train}}(g=1) P_{\text{train}}(m=1)],
        \end{aligned}
        $$
        and
        $$
        \begin{aligned}
            \Delta_{gm} &= \begin{cases}
            P_{\text{test}}(\hat{m}=1 \vert g=1) - P_{\text{train}}(m=1 \vert g=1) \\ \text{if measuring }G\rightarrow M\\
            P_{\text{test}}(\hat{g}=1 \vert m=1) - P_{\text{train}}(g=1 \vert m=1) \\ \text{if measuring }M\rightarrow G
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
    <h2>Comparison with existing metrics</h2>
    {% katexmm %}
    <p>Our proposed metrics have three advantages over existing single-attribute
        co-occurence-based metrics.</p>
    <p><b>(Advantage 1) Our metrics account for co-occurrences with multiple attributes</b></p>
    <p class="mt20">
        If a model, for example,
        learns the combination of $a_1$ and $a_2$, i.e., $m=\{a_1, a_2\}\in\mathcal{M}$, are correlated with
        $g\in\mathcal{G}$, it can exploit this correlation, potentially leading to bias amplification. By iterating over all $m \in \mathcal{M},$ our proposed metric accounts for amplification from
        single and multiple attributes. Thus, capturing sets of attributes exhibiting amplification, which are not
        accounted for by existing metrics.
    </p>
    {% endkatexmm %}
    <p><b>(Advantage 2) Negative and positive values do not cancel each other out</b></p>
    <p class="mt20">
        {% katexmm %}
        Existing metrics calculate bias amplification by aggregating over the difference in bias scores for each
        attribute individually. Suppose there is a dataset with two annotated attributes $a_1$ and $a_2$. It is possible
        that ${\Delta_{ga_1} \approx -\Delta_{ga_2}}$. Since our metrics use absolute values, we ensure that positive and negative bias amplifications per attribute do
        not cancel each other out. </p>
    {% endkatexmm %}
    <p><b>(Advantage 3) Our metrics are more interpretable</b></p>
    <p class="mt20">
        {% katexmm %}
        There is a lack of intuition as to the &ldquo;ideal&rdquo; bias amplification value. One interpretation is that smaller values are more desirable. This becomes less clear when values are negative, as occurs in several bias mitigation works{%cite wang2020fair ramaswamy2020debiasing%}. Negative bias amplification indicates bias in the predictions is in the opposite direction than that in the training set. However, this is not always ideal. First, there often exists a trade-off between performance and smaller bias amplification values. Second, high magnitude negative bias amplification may lead to erasure of certain groups. For example, in imSitu, $$\text{bias}_{\text{train}}(\{\text{typing}\}, \text{female}) = \text{0.52}.$$ Negative bias amplification signifies the model underpredicts $(\{\text{typing}\}, \text{female})$, which could reinforce negative gender stereotypes{% cite zhao2017mals %}.
        {% endkatexmm %}
    </p>
    <p>
    Instead, we may want to minimize the distance between the bias amplification value and zero. This interpretation offers the advantage that large negative values are also not desirable. However, a potential dilemma occurs when interpreting two values with the same magnitude but opposite signs, which is a value-laden decision and depends on the system's context. Additionally, under this alternative interpretation, Advantage 2 becomes more pressing as this suggests we are interpreting models as less biased than they are in practice.
    </p>
    <p>
    Our proposed metrics are easy to interpret. Since we use absolute differences, the ideal value is unambiguously zero. Further, reporting variance provides intuition as to whether amplification is uniform across all attribute-group pairs or if particular pairs are more amplified.
    </p>
</div>

{% if site.scholar.bibliography %}
<div class="inner" style="padding-top: 80px;">
  <div class="reference-container">
    <div class="reference-title">
      <p><b>References</b></p>
    </div>
    <div class="reference-text">
      {% bibliography --cited_in_order %}
    </div>
  </div>
</div>
{% endif %}

{% if page.authors != acknowledgments %}
<div class="inner" style="padding-top: 80px;">
  <div class="acknowledge-container">
    <div class="acknowledge-title">
      <p><b>Acknowledgments</b></p>
    </div>
    <div class="acknowledge-text">
      <p>{{ page.acknowledgments }}</p>
    </div>
  </div>
</div>
{% endif %}