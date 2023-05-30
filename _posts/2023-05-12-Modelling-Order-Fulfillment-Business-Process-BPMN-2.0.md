---
title: Modelling Order Fulfillment Business Process with BPMN 2.0
tags: [Business Process Modelling, Process Management, Order fulfillment, BPMN 2.0, AWS/MWS, ERP, SQL, XML]
style: fill
color: warning
description: Documenting and demonstrating my process-modelling and -management knowledge and the skills to document business processes with BPMN 2.0 at my own Order Fulfillment Business Process conceived and implemented in PHP code. 
---

<style type="text/css">
 .markdown-body img:not(.emoji) {
    display: block;
    max-width: 1349px; !important
}
</style>


For my company ZS Trading I had modeled my own Business Process for Order Fulfillment with the `Fulfillment by Amazon (FBA)` service, which I had conceived and specified along the enabling possiblities of Amazon Market Web Service (MWS) a few years ago.

For this purpose I applied my BPMN 2.0 skills in the business process modelling, which is short for the Business Process Model and Notation standard which is an ISO standard for nearly 10 years already.


![Order Fulfillment Process in BPMN 2.0](/blog/modelling-business-process/ZST-Order-Fufillment-Process-bpmn20.png "Order Fulfillment Process in BPMN 2.0"){: width="1346" }

The Process starts with the compliation of customer orders from the ERP system that were earmarked by a meta process to be fulfilled with the FBA service in form of an XML file.

In a second step those order positions part numbers are collected and queried against existing SKUs in a MySQL database, which in return will be matched with live inventory data that is available for fulfillment right away along with inventory health data (storage durations).

As a last step shown in more detail is the process step `Decide on SKUs for fulfillment` which list all thta data and enables the user to take an informed decision.

Sample Screen from Step two of the process `Query SKUs` towards `Decide on SKUs for fulfillment`:

!["Order Fulfillment Process Screen Output Example"](/blog/modelling-business-process/ZS-Trading-Order-Fulfillment-Process-MWS-FBA-Screen-blackbg.png "Order Fulfillment Process Screen Output")

Below is the process implemented in PHP 7 code which shows the relevant functions and processes with code omitted for better readability. The part of the code is shown where SKUs get queried for part number to be fulfilled in a two step drilled down process: 

First part is the SQL query which skus belong to a part number which are to be collected.
Second part is the query of SKUs with Amazon's Market Web Service (MWS) and Fulfillment by Amazon (FBA) live inventory data via REST API.

In the screen above the user has the choice between all the SKUs listed, where as only SKU X10-2018-DE has got stock ready to be shipped. This corresponds to the process step `Decide on SKUs for fulfillment`.


```python
from IPython.display import HTML
HTML(filename="/blog/modelling-business-process/Fulfillment-Code.html")
```



<pre>
 <code id="htmlViewer" style="color:rgb(248, 248, 242); font-weight:400;background-color:rgb(43, 43, 43);background:rgb(43, 43, 43);display:block;padding: .5em;"><span style="color:rgb(245, 171, 53); font-weight:400;">&lt;?php</span>

<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// ............</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>

<span style="color:rgb(255, 160, 122); font-weight:400;">$serviceUrl2</span> =
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;https://mws-eu.amazonservices.com/FulfillmentOutboundShipment/2010-10-01&quot;</span>;
<span style="color:rgb(255, 160, 122); font-weight:400;">$serviceUrl1</span> =
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;https://mws-eu.amazonservices.com/FulfillmentInventory/2010-10-01&quot;</span>;

<span style="color:rgb(255, 160, 122); font-weight:400;">$config2</span> = [
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ServiceURL&quot;</span> =&gt; <span style="color:rgb(255, 160, 122); font-weight:400;">$serviceUrl2</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ProxyHost&quot;</span> =&gt; <span style="color:rgb(245, 171, 53); font-weight:400;">null</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ProxyPort&quot;</span> =&gt; -<span style="color:rgb(245, 171, 53); font-weight:400;">1</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ProxyUsername&quot;</span> =&gt; <span style="color:rgb(245, 171, 53); font-weight:400;">null</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ProxyPassword&quot;</span> =&gt; <span style="color:rgb(245, 171, 53); font-weight:400;">null</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;MaxErrorRetry&quot;</span> =&gt; <span style="color:rgb(245, 171, 53); font-weight:400;">3</span>,
];
<span style="color:rgb(255, 160, 122); font-weight:400;">$config1</span> = [
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ServiceURL&quot;</span> =&gt; <span style="color:rgb(255, 160, 122); font-weight:400;">$serviceUrl1</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ProxyHost&quot;</span> =&gt; <span style="color:rgb(245, 171, 53); font-weight:400;">null</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ProxyPort&quot;</span> =&gt; -<span style="color:rgb(245, 171, 53); font-weight:400;">1</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ProxyUsername&quot;</span> =&gt; <span style="color:rgb(245, 171, 53); font-weight:400;">null</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ProxyPassword&quot;</span> =&gt; <span style="color:rgb(245, 171, 53); font-weight:400;">null</span>,
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;MaxErrorRetry&quot;</span> =&gt; <span style="color:rgb(245, 171, 53); font-weight:400;">3</span>,
];

<span style="color:rgb(255, 160, 122); font-weight:400;">$service2</span> = <span style="color:rgb(220, 198, 224); font-weight:400;">new</span> <span class="hljs-title class_">FBAOutboundServiceMWS_Client</span>(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    <span style="color:rgb(255, 160, 122); font-weight:400;">$config2</span>,
    APPLICATION_NAME,
    APPLICATION_VERSION
);
<span style="color:rgb(255, 160, 122); font-weight:400;">$service1</span> = <span style="color:rgb(220, 198, 224); font-weight:400;">new</span> <span class="hljs-title class_">FBAInventoryServiceMWS_Client</span>(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    <span style="color:rgb(255, 160, 122); font-weight:400;">$config1</span>,
    APPLICATION_NAME,
    APPLICATION_VERSION
);

<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// ............</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>

<span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;h4&gt;ERP order no # &quot;</span> .
    <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;SellerFulfillmentOrderId&quot;</span>] .
    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;/h4&gt;&quot;</span>;

<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// ............</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>

<span style="color:rgb(220, 198, 224); font-weight:400;">for</span> (<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span> = <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>; <span style="color:rgb(255, 160, 122); font-weight:400;">$z</span> &lt; <span class="hljs-title function_ invoke__">count</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;tBestellung&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$x</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;twarenkorbpos&quot;</span>]); <span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>++) {
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;pos count: &quot;</span> . <span class="hljs-title function_ invoke__">count</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;tBestellung&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$x</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;twarenkorbpos&quot;</span>]);
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;div id=&#x27;expand&#x27;&gt;&quot;</span>;
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;pre&gt;&quot;</span>;
    <span class="hljs-title function_ invoke__">print_r</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span>);
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;/pre&gt;&quot;</span>;
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;/div&gt;&quot;</span>;

    <span style="color:rgb(255, 160, 122); font-weight:400;">$ignoredartnos</span> = [<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;0001&quot;</span>, <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;00001&quot;</span>, <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;0002&quot;</span>, <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;00002&quot;</span>, <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;1162&quot;</span>];
    <span style="color:rgb(255, 160, 122); font-weight:400;">$tobecheckedartno</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;tBestellung&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$x</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;twarenkorbpos&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;cArtNr&quot;</span>];

    <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (
        <span class="hljs-title function_ invoke__">in_array</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$tobecheckedartno</span>, <span style="color:rgb(255, 160, 122); font-weight:400;">$ignoredartnos</span>) ||
        <span class="hljs-title function_ invoke__">mb_substr</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$tobecheckedartno</span>, <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>, <span style="color:rgb(245, 171, 53); font-weight:400;">2</span>) == <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;90&quot;</span>
    ) {
        <span style="color:rgb(212, 208, 171); font-weight:400;">// only real items!!</span>
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Order line item ID: &quot;</span> .
            (<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span> + <span style="color:rgb(245, 171, 53); font-weight:400;">1</span>) .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; with artno  &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$tobecheckedartno</span> .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; is discarded.&lt;br&gt;########################
							&lt;br&gt;&quot;</span>;
        <span style="color:rgb(220, 198, 224); font-weight:400;">continue</span>;
    }

    <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (
        <span style="color:rgb(220, 198, 224); font-weight:400;">isset</span>(
            <span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;tBestellung&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$x</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;twarenkorbpos&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;twarenkorbposeigenschaft&quot;</span>
            ]
        )
    ) {
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;artno&quot;</span>] =
            <span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;tBestellung&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$x</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;twarenkorbpos&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;twarenkorbposeigenschaft&quot;</span>
            ][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;cArtNr&quot;</span>];
        <span style="color:rgb(255, 160, 122); font-weight:400;">$sql</span> =
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;SELECT sku FROM &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$tname</span> .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; WHERE vanr = &#x27;&quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;artno&quot;</span>] .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&#x27;&quot;</span>;
        <span style="color:rgb(212, 208, 171); font-weight:400;">//echo $sql;</span>
    } <span style="color:rgb(220, 198, 224); font-weight:400;">else</span> {
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;artno&quot;</span>] =
            <span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;tBestellung&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$x</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;twarenkorbpos&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;cArtNr&quot;</span>];
        <span style="color:rgb(255, 160, 122); font-weight:400;">$sql</span> =
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;SELECT sku FROM &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$tname</span> .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; WHERE artnr = &#x27;&quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;artno&quot;</span>] .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&#x27;&quot;</span>;
    }

    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Order line item ID: &quot;</span> . (<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span> + <span style="color:rgb(245, 171, 53); font-weight:400;">1</span>) . <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt; &quot;</span>;

    <span style="color:rgb(212, 208, 171); font-weight:400;">//query fba-sku for the artno (reverse mapping)</span>

    <span style="color:rgb(255, 160, 122); font-weight:400;">$q</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$pdo</span>-&gt;<span class="hljs-title function_ invoke__">query</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$sql</span>);
    <span style="color:rgb(255, 160, 122); font-weight:400;">$q</span>-&gt;<span class="hljs-title function_ invoke__">setFetchMode</span>(PDO::<span class="hljs-variable constant_">FETCH_ASSOC</span>);
    <span style="color:rgb(255, 160, 122); font-weight:400;">$skuresult</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$q</span>-&gt;<span class="hljs-title function_ invoke__">fetchAll</span>();

    <span style="color:rgb(212, 208, 171); font-weight:400;">// no FBA sku found</span>
    <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (<span style="color:rgb(255, 160, 122); font-weight:400;">$q</span>-&gt;<span class="hljs-title function_ invoke__">rowCount</span>() == <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>) {
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;span style=&#x27;color:red;&#x27;&gt;&lt;b&gt;No FBA SKU for &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;artno&quot;</span>] .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; found! A complete fulfillment order is acc. to mapping data not possible. If a FBA SKU for this article exists, it needs to be maintained in the mapping Database.&lt;/b&gt;&lt;/span&gt;&lt;br&gt;########################
						&lt;br&gt;&quot;</span>;
        <span style="color:rgb(220, 198, 224); font-weight:400;">continue</span>;
    }

    <span style="color:rgb(255, 160, 122); font-weight:400;">$skuresultflat</span> = <span class="hljs-title function_ invoke__">array_column</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$skuresult</span>, <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku&quot;</span>);
    <span style="color:rgb(255, 160, 122); font-weight:400;">$skuresultflat</span> = <span class="hljs-title function_ invoke__">array_flatten</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$skuresultflat</span>);
    <span style="color:rgb(255, 160, 122); font-weight:400;">$skuresultflat</span> = <span class="hljs-title function_ invoke__">array_values</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$skuresultflat</span>);

    <span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>
    <span style="color:rgb(212, 208, 171); font-weight:400;">// ............</span>
    <span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>

    <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>] = <span style="color:rgb(255, 160, 122); font-weight:400;">$skuresult</span>;

    <span style="color:rgb(212, 208, 171); font-weight:400;">// get fba stock for each sku through MWS REST API</span>
    <span style="color:rgb(255, 160, 122); font-weight:400;">$request</span> = <span style="color:rgb(220, 198, 224); font-weight:400;">new</span> <span class="hljs-title class_">FBAInventoryServiceMWS_Model_ListInventorySupplyRequest</span>();
    <span style="color:rgb(255, 160, 122); font-weight:400;">$request</span>-&gt;<span class="hljs-title function_ invoke__">setSellerId</span>(MERCHANT_ID);
    <span style="color:rgb(255, 160, 122); font-weight:400;">$request</span>-&gt;<span class="hljs-title function_ invoke__">setMarketplace</span>(MARKETPLACE_ID);

    <span style="color:rgb(255, 160, 122); font-weight:400;">$skus</span> = <span style="color:rgb(220, 198, 224); font-weight:400;">new</span> <span class="hljs-title class_">FBAInventoryServiceMWS_Model_SellerSkuList</span>();
    <span style="color:rgb(255, 160, 122); font-weight:400;">$skus</span>-&gt;<span class="hljs-title function_ invoke__">setmember</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$skuresultflat</span>);
    <span style="color:rgb(255, 160, 122); font-weight:400;">$request</span>-&gt;<span class="hljs-title function_ invoke__">setSellerSkus</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$skus</span>);

    <span style="color:rgb(212, 208, 171); font-weight:400;">// object or array of parameters</span>
    <span class="hljs-title function_ invoke__">invokeListInventorySupply</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$service1</span>, <span style="color:rgb(255, 160, 122); font-weight:400;">$request</span>);
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;inventory call finsihed&quot;</span>;

    <span style="color:rgb(212, 208, 171); font-weight:400;">// qty</span>
    <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;qty&quot;</span>] = <span class="hljs-title function_ invoke__">round</span>(
        <span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;tBestellung&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$x</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;twarenkorbpos&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;fAnzahl&quot;</span>],
        <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>
    );
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt;&lt;b&gt;Ordered Quantity: &quot;</span> .
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;qty&quot;</span>] .
        <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;/b&gt;&lt;br&gt;&quot;</span>;

    <span style="color:rgb(212, 208, 171); font-weight:400;">//SellerFulfillmentOrderItemId</span>
    <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][
        <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;SellerFulfillmentOrderItemId&quot;</span>
    ] = <span style="color:rgb(255, 160, 122); font-weight:400;">$erporder</span> . <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;-&quot;</span> . <span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>;

    <span style="color:rgb(212, 208, 171); font-weight:400;">//RNo check</span>
    <span style="color:rgb(255, 160, 122); font-weight:400;">$RNo</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$order</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;@attributes&quot;</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;cRechnungsNr&quot;</span>];
    <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (<span style="color:rgb(255, 160, 122); font-weight:400;">$RNo</span> == <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&quot;</span>) {
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;h3 style=&#x27;color:red;&#x27;&gt;Warning: No Invoice Number detected! Invoice created at all for &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$auftragsno</span> .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;? &lt;b&gt;Strictly&lt;/b&gt; no Fufillment-Order allowed without created ERP invoice!&lt;/h3&gt;&quot;</span>;
    }

    <span style="color:rgb(212, 208, 171); font-weight:400;">// Form 2 mainpart- SKU Selection</span>

    <span style="color:rgb(212, 208, 171); font-weight:400;">//echo &quot;ERP Auftrag #:&quot;. $_SESSION[$sessionstring][&#x27;SellerFulfillmentOrderId&#x27;].&quot;&lt;br&gt;&quot;;</span>
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Ext bestell #:&quot;</span> .
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;DisplayableOrderId&quot;</span>] .
        <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt;&quot;</span>;
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Lieferadresse Name #:&quot;</span> .
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;DestinationAddress.Name&quot;</span>] .
        <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt;&quot;</span>;
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;artno/vanr: &lt;b&gt;&quot;</span> .
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;artno&quot;</span>] .
        <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;/b&gt;: &lt;br&gt;&quot;</span>;
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; Choose FBA sku. LIVE FBA DATA[sku | total | instock | availability]:&lt;select name=&#x27;&quot;</span> .
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;SellerFulfillmentOrderItemId&quot;</span>
        ] .
        <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&#x27; id=&#x27;&quot;</span> .
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;SellerFulfillmentOrderItemId&quot;</span>
        ] .
        <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&#x27;&gt;&quot;</span>;

    <span class="hljs-title function_ invoke__">sort_array_of_array</span>(
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>],
        <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;InStockSupplyQuantity&quot;</span>
    );
    <span style="color:rgb(212, 208, 171); font-weight:400;">// skus iteration</span>
    <span style="color:rgb(220, 198, 224); font-weight:400;">for</span> (
        <span style="color:rgb(255, 160, 122); font-weight:400;">$y</span> = <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>;
        <span style="color:rgb(255, 160, 122); font-weight:400;">$y</span> &lt; <span class="hljs-title function_ invoke__">count</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>]);
        <span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>++
    ) {
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;option&gt;&quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku&quot;</span>
            ] .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;TotalSupplyQuantity&quot;</span>
            ] .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;InStockSupplyQuantity&quot;</span>
            ] .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;EarliestAvailability&quot;</span>
            ] .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;/option&gt;&quot;</span>;
    }
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;/select&gt;&lt;br&gt;&quot;</span>;

    <span style="color:rgb(212, 208, 171); font-weight:400;">// screen output for user</span>
    <span style="color:rgb(212, 208, 171); font-weight:400;">// print Inventory health information for each sku for optimized choice of fulfillment sku</span>
    <span style="color:rgb(212, 208, 171); font-weight:400;">// inventory health report array</span>
    <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span> = [];
    <span style="color:rgb(255, 160, 122); font-weight:400;">$fp</span> = <span class="hljs-title function_ invoke__">fopen</span>(<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;DE-invent-health.txt&quot;</span>, <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;r&quot;</span>);
    <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> ((<span style="color:rgb(255, 160, 122); font-weight:400;">$headers</span> = <span class="hljs-title function_ invoke__">fgetcsv</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$fp</span>, <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>, <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\t&quot;</span>)) !== <span style="color:rgb(245, 171, 53); font-weight:400;">false</span>) {
        <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (<span style="color:rgb(255, 160, 122); font-weight:400;">$headers</span>) {
            <span style="color:rgb(220, 198, 224); font-weight:400;">while</span> ((<span style="color:rgb(255, 160, 122); font-weight:400;">$line</span> = <span class="hljs-title function_ invoke__">fgetcsv</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$fp</span>, <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>, <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\t&quot;</span>)) !== <span style="color:rgb(245, 171, 53); font-weight:400;">false</span>) {
                <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (<span style="color:rgb(255, 160, 122); font-weight:400;">$line</span>) {
                    <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (<span class="hljs-title function_ invoke__">sizeof</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$line</span>) == <span class="hljs-title function_ invoke__">sizeof</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$headers</span>)) {
                        <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>[] = <span class="hljs-title function_ invoke__">array_combine</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$headers</span>, <span style="color:rgb(255, 160, 122); font-weight:400;">$line</span>);
                    }
                }
            };
        }
    }
    <span class="hljs-title function_ invoke__">fclose</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$fp</span>);
    <span style="color:rgb(212, 208, 171); font-weight:400;">//echo &#x27;&lt;pre&gt;&#x27; . var_export($IHreport, true) . &#x27;&lt;/pre&gt;&#x27;;die;</span>

    <span style="color:rgb(220, 198, 224); font-weight:400;">for</span> (
        <span style="color:rgb(255, 160, 122); font-weight:400;">$y</span> = <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>;
        <span style="color:rgb(255, 160, 122); font-weight:400;">$y</span> &lt; <span class="hljs-title function_ invoke__">count</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>]);
        <span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>++
    ) {
        <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf12feeqty</span> = <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&quot;</span>;
        <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf6feeqty</span> = <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&quot;</span>;
        <span style="color:rgb(255, 160, 122); font-weight:400;">$age0</span> = <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&quot;</span>;
        <span style="color:rgb(255, 160, 122); font-weight:400;">$age91</span> = <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&quot;</span>;
        <span style="color:rgb(255, 160, 122); font-weight:400;">$age181</span> = <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&quot;</span>;
        <span style="color:rgb(255, 160, 122); font-weight:400;">$age271</span> = <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&quot;</span>;
        <span style="color:rgb(255, 160, 122); font-weight:400;">$age365</span> = <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&quot;</span>;

        <span style="color:rgb(220, 198, 224); font-weight:400;">for</span> (<span style="color:rgb(255, 160, 122); font-weight:400;">$i</span> = <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>; <span style="color:rgb(255, 160, 122); font-weight:400;">$i</span> &lt; <span class="hljs-title function_ invoke__">count</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>); <span style="color:rgb(255, 160, 122); font-weight:400;">$i</span>++) {
            <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (
                <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$i</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku&quot;</span>] ==
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku&quot;</span>
                ]
            ) {
                <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf12feeqty</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$i</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;qty-to-be-charged-ltsf-12-mo&quot;</span>];
                <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf6feeqty</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$i</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;qty-to-be-charged-ltsf-6-mo&quot;</span>];
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age0</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$i</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;inv-age-0-to-90-days&quot;</span>];
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age91</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$i</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;inv-age-91-to-180-days&quot;</span>];
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age181</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$i</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;inv-age-181-to-270-days&quot;</span>];
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age271</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$i</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;inv-age-271-to-365-days&quot;</span>];
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age365</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$IHreport</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$i</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;inv-age-365-plus-days&quot;</span>];
            }
        }
        <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (
            <span class="hljs-title function_ invoke__">in_array</span>(
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku&quot;</span>
                ],
                <span style="color:rgb(255, 160, 122); font-weight:400;">$excessSKU</span>
            )
        ) {
            <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt;&lt;span style=&#x27;color:red;&#x27;&gt;&quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;TotalSupplyQuantity&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;InStockSupplyQuantity&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;EarliestAvailability&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | Excess SKU!&lt;/span&gt;&quot;</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | LTF 12 months fee qty: &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf12feeqty</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | LTF 6 months fee qty: &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf6feeqty</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | Age (90|180|270|365|365+) qty: &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age0</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age91</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age181</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age271</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age365</span>;
        } <span style="color:rgb(220, 198, 224); font-weight:400;">elseif</span> (
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku&quot;</span>
            ] == <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;DE-WP444-FBA&quot;</span>
        ) {
            <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt;&lt;span style=&#x27;color:blue;&#x27;&gt;&quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;TotalSupplyQuantity&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;InStockSupplyQuantity&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;EarliestAvailability&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | Sonderfall SKU&lt;/span&gt;&quot;</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | LTF 12 months fee qty: &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf12feeqty</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | LTF 6 months fee qty: &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf6feeqty</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | Age (90|180|270|365|365+) qty: &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age0</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age91</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age181</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age271</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age365</span>;
        } <span style="color:rgb(220, 198, 224); font-weight:400;">else</span> {
            <span style="color:rgb(212, 208, 171); font-weight:400;">// regular sku</span>
            <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt;&quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;TotalSupplyQuantity&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;InStockSupplyQuantity&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;EarliestAvailability&quot;</span>
                ] .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | LTF 12 months fee qty: &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf12feeqty</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | LTF 6 months fee qty: &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$ltf6feeqty</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | Age (90|180|270|365|365+) qty: &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age0</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age91</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age181</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age271</span> .
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; | &quot;</span> .
                <span style="color:rgb(255, 160, 122); font-weight:400;">$age365</span>;
        }
    }
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt;&quot;</span> .
        <span class="hljs-title function_ invoke__">count</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>]) .
        <span style="color:rgb(171, 227, 56); font-weight:400;">&quot; distinct FBA SKUs for &quot;</span> .
        <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;artno&quot;</span>];
    <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt;########################&lt;br&gt;&quot;</span>;
}
<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// ............</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>

<span style="color:rgb(212, 208, 171); font-weight:400;">// MWS SDK functions</span>

<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// ............</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>

<span style="color:rgb(248, 248, 242); font-weight:400;"><span style="color:rgb(220, 198, 224); font-weight:400;">function</span> <span style="color:rgb(0, 224, 224); font-weight:400;">invokeListInventorySupply</span>(<span style="color:rgb(245, 171, 53); font-weight:400;">
    FBAInventoryServiceMWS_Interface <span style="color:rgb(255, 160, 122); font-weight:400;">$service</span>,
    <span style="color:rgb(255, 160, 122); font-weight:400;">$request</span>
</span>) </span>{
    <span style="color:rgb(212, 208, 171); font-weight:400;">//$x lineitemid</span>
    <span style="color:rgb(220, 198, 224); font-weight:400;">try</span> {
        <span style="color:rgb(255, 160, 122); font-weight:400;">$response</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$service</span>-&gt;<span class="hljs-title function_ invoke__">ListInventorySupply</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$request</span>);

        <span style="color:rgb(212, 208, 171); font-weight:400;">// use global values outside this function</span>
        <span style="color:rgb(220, 198, 224); font-weight:400;">global</span> <span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>, <span style="color:rgb(255, 160, 122); font-weight:400;">$erporder</span>, <span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>;
        <span style="color:rgb(212, 208, 171); font-weight:400;">//echo &quot;x: &quot;.$x.&quot; ERPorder variable: &quot;.$erporder.&quot;&lt;br&gt;&quot;;</span>
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sessionstring: &quot;</span> . <span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span> . <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;&lt;br&gt;&quot;</span>;

        <span style="color:rgb(212, 208, 171); font-weight:400;">//echo (&quot;Service Response\n&quot;);</span>
        <span style="color:rgb(212, 208, 171); font-weight:400;">//echo (&quot;=============================================================================\n&quot;);</span>

        <span style="color:rgb(255, 160, 122); font-weight:400;">$xmlstring</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$response</span>-&gt;<span class="hljs-title function_ invoke__">toXML</span>();
        <span style="color:rgb(212, 208, 171); font-weight:400;">//Three line xml2array:</span>
        <span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span> = <span class="hljs-title function_ invoke__">simplexml_load_string</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$xmlstring</span>);
        <span style="color:rgb(255, 160, 122); font-weight:400;">$json</span> = <span class="hljs-title function_ invoke__">json_encode</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$xml</span>);
        <span style="color:rgb(255, 160, 122); font-weight:400;">$array</span> = <span class="hljs-title function_ invoke__">json_decode</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$json</span>, <span style="color:rgb(245, 171, 53); font-weight:400;">true</span>);

        <span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span> = <span style="color:rgb(255, 160, 122); font-weight:400;">$array</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ListInventorySupplyResult&quot;</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;InventorySupplyList&quot;</span>];

        <span style="color:rgb(212, 208, 171); font-weight:400;">// normalize amzn api array</span>
        <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (!<span style="color:rgb(220, 198, 224); font-weight:400;">isset</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>][<span style="color:rgb(245, 171, 53); font-weight:400;">0</span>])) {
            <span style="color:rgb(212, 208, 171); font-weight:400;">// list has only single sku</span>
            <span style="color:rgb(255, 160, 122); font-weight:400;">$rebuild</span> = [];

            <span style="color:rgb(212, 208, 171); font-weight:400;">//Check to see if &#x27;properties&#x27; is only one, if it</span>
            <span style="color:rgb(212, 208, 171); font-weight:400;">//is then wrap it in an array of its own.</span>

            <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (
                <span class="hljs-title function_ invoke__">is_array</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>]) &amp;&amp;
                !<span style="color:rgb(220, 198, 224); font-weight:400;">isset</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>][<span style="color:rgb(245, 171, 53); font-weight:400;">0</span>])
            ) {
                <span style="color:rgb(212, 208, 171); font-weight:400;">//Only one propery found, wrap it in an array</span>
                <span style="color:rgb(255, 160, 122); font-weight:400;">$rebuild</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>] = [<span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>]];
            }

            <span style="color:rgb(212, 208, 171); font-weight:400;">//echo &#x27;################################################&#x27;;</span>
            <span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>] = <span style="color:rgb(255, 160, 122); font-weight:400;">$rebuild</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>];
            <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;sku list member normalized &lt;br&gt;&quot;</span>;
        }

        <span style="color:rgb(212, 208, 171); font-weight:400;">//foreach($inventory as $member){</span>
        <span style="color:rgb(220, 198, 224); font-weight:400;">for</span> (<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span> = <span style="color:rgb(245, 171, 53); font-weight:400;">0</span>; <span style="color:rgb(255, 160, 122); font-weight:400;">$y</span> &lt; <span class="hljs-title function_ invoke__">count</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>]); <span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>++) {
            <span style="color:rgb(212, 208, 171); font-weight:400;">// add new array for line item [&#x27;Items.member&#x27;][]</span>
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;TotalSupplyQuantity&quot;</span>
            ] = <span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;TotalSupplyQuantity&quot;</span>];
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;InStockSupplyQuantity&quot;</span>
            ] = <span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;InStockSupplyQuantity&quot;</span>];
            <span style="color:rgb(255, 160, 122); font-weight:400;">$_SESSION</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$sessionstring</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Items.member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$z</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;allskus&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][
                <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;EarliestAvailability&quot;</span>
            ] =
                <span style="color:rgb(255, 160, 122); font-weight:400;">$inventory</span>[<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;member&quot;</span>][<span style="color:rgb(255, 160, 122); font-weight:400;">$y</span>][<span style="color:rgb(171, 227, 56); font-weight:400;">&quot;EarliestAvailability&quot;</span>][
                    <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;TimepointType&quot;</span>
                ];
        }

        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ResponseHeaderMetadata: &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$response</span>-&gt;<span class="hljs-title function_ invoke__">getResponseHeaderMetadata</span>() .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\n&quot;</span>;
    } <span style="color:rgb(220, 198, 224); font-weight:400;">catch</span> (FBAInventoryServiceMWS_Exception <span style="color:rgb(255, 160, 122); font-weight:400;">$ex</span>) {
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Caught Exception: &quot;</span> . <span style="color:rgb(255, 160, 122); font-weight:400;">$ex</span>-&gt;<span class="hljs-title function_ invoke__">getMessage</span>() . <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\n&quot;</span>;
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Response Status Code: &quot;</span> . <span style="color:rgb(255, 160, 122); font-weight:400;">$ex</span>-&gt;<span class="hljs-title function_ invoke__">getStatusCode</span>() . <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\n&quot;</span>;
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Error Code: &quot;</span> . <span style="color:rgb(255, 160, 122); font-weight:400;">$ex</span>-&gt;<span class="hljs-title function_ invoke__">getErrorCode</span>() . <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\n&quot;</span>;
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Error Type: &quot;</span> . <span style="color:rgb(255, 160, 122); font-weight:400;">$ex</span>-&gt;<span class="hljs-title function_ invoke__">getErrorType</span>() . <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\n&quot;</span>;
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;Request ID: &quot;</span> . <span style="color:rgb(255, 160, 122); font-weight:400;">$ex</span>-&gt;<span class="hljs-title function_ invoke__">getRequestId</span>() . <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\n&quot;</span>;
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;XML: &quot;</span> . <span style="color:rgb(255, 160, 122); font-weight:400;">$ex</span>-&gt;<span class="hljs-title function_ invoke__">getXML</span>() . <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\n&quot;</span>;
        <span style="color:rgb(220, 198, 224); font-weight:400;">echo</span> <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;ResponseHeaderMetadata: &quot;</span> .
            <span style="color:rgb(255, 160, 122); font-weight:400;">$ex</span>-&gt;<span class="hljs-title function_ invoke__">getResponseHeaderMetadata</span>() .
            <span style="color:rgb(171, 227, 56); font-weight:400;">&quot;\n&quot;</span>;
    }
    <span style="color:rgb(220, 198, 224); font-weight:400;">return</span>;
}
<span style="color:rgb(212, 208, 171); font-weight:400;">/// #######################</span>

<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// ............</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>

<span style="color:rgb(248, 248, 242); font-weight:400;"><span style="color:rgb(220, 198, 224); font-weight:400;">function</span> <span style="color:rgb(0, 224, 224); font-weight:400;">array_flatten</span>(<span style="color:rgb(245, 171, 53); font-weight:400;"><span style="color:rgb(255, 160, 122); font-weight:400;">$array</span> = <span style="color:rgb(245, 171, 53); font-weight:400;">null</span></span>)
</span>{
    <span style="color:rgb(255, 160, 122); font-weight:400;">$result</span> = [];

    <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (!<span class="hljs-title function_ invoke__">is_array</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$array</span>)) {
        <span style="color:rgb(255, 160, 122); font-weight:400;">$array</span> = <span class="hljs-title function_ invoke__">func_get_args</span>();
    }

    <span style="color:rgb(220, 198, 224); font-weight:400;">foreach</span> (<span style="color:rgb(255, 160, 122); font-weight:400;">$array</span> <span style="color:rgb(220, 198, 224); font-weight:400;">as</span> <span style="color:rgb(255, 160, 122); font-weight:400;">$key</span> =&gt; <span style="color:rgb(255, 160, 122); font-weight:400;">$value</span>) {
        <span style="color:rgb(220, 198, 224); font-weight:400;">if</span> (<span class="hljs-title function_ invoke__">is_array</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$value</span>)) {
            <span style="color:rgb(255, 160, 122); font-weight:400;">$result</span> = <span class="hljs-title function_ invoke__">array_merge</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$result</span>, <span class="hljs-title function_ invoke__">array_flatten</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$value</span>));
        } <span style="color:rgb(220, 198, 224); font-weight:400;">else</span> {
            <span style="color:rgb(255, 160, 122); font-weight:400;">$result</span> = <span class="hljs-title function_ invoke__">array_merge</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$result</span>, [<span style="color:rgb(255, 160, 122); font-weight:400;">$key</span> =&gt; <span style="color:rgb(255, 160, 122); font-weight:400;">$value</span>]);
        }
    }

    <span style="color:rgb(220, 198, 224); font-weight:400;">return</span> <span style="color:rgb(255, 160, 122); font-weight:400;">$result</span>;
}
<span style="color:rgb(248, 248, 242); font-weight:400;"><span style="color:rgb(220, 198, 224); font-weight:400;">function</span> <span style="color:rgb(0, 224, 224); font-weight:400;">sort_array_of_array</span>(<span style="color:rgb(245, 171, 53); font-weight:400;">&amp;<span style="color:rgb(255, 160, 122); font-weight:400;">$array</span>, <span style="color:rgb(255, 160, 122); font-weight:400;">$subfield</span></span>)
</span>{
    <span style="color:rgb(255, 160, 122); font-weight:400;">$sortarray</span> = [];
    <span style="color:rgb(220, 198, 224); font-weight:400;">foreach</span> (<span style="color:rgb(255, 160, 122); font-weight:400;">$array</span> <span style="color:rgb(220, 198, 224); font-weight:400;">as</span> <span style="color:rgb(255, 160, 122); font-weight:400;">$key</span> =&gt; <span style="color:rgb(255, 160, 122); font-weight:400;">$row</span>) {
        <span style="color:rgb(255, 160, 122); font-weight:400;">$sortarray</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$key</span>] = <span style="color:rgb(255, 160, 122); font-weight:400;">$row</span>[<span style="color:rgb(255, 160, 122); font-weight:400;">$subfield</span>];
    }

    <span class="hljs-title function_ invoke__">array_multisort</span>(<span style="color:rgb(255, 160, 122); font-weight:400;">$sortarray</span>, SORT_DESC, <span style="color:rgb(255, 160, 122); font-weight:400;">$array</span>);
}

<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// ............</span>
<span style="color:rgb(212, 208, 171); font-weight:400;">// code omitted</span>

<span style="color:rgb(245, 171, 53); font-weight:400;">?&gt;</span>
</code></pre>




```python

```
