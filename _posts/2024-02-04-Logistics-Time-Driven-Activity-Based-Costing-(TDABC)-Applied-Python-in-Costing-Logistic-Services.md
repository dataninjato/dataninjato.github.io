---
title: Logistics Time-Driven Activity-Based Costing (TDABC) - Applied Python in Costing Logistic Services
tags: [Activity-Based Costing, Time-Driven Activity-Based Costing, Python, Supply Chain, Logistics]
style: fill
color: secondary
description: Implementing Time-Driven Activity-Based Costing (TDABC) with Python and extending with storage costs. Thus significantly enhancing logistics decision-making by providing valuable insights and improving various facets of operational efficiency and profitability management.
---
<style type="text/css">
 .markdown-body img:not(.emoji) {
    display: block;
    max-width: 1349px; !important
}
</style>

Time-Driven Activity-Based Costing (TDABC) employs time as a fundamental factor in cost determination, leveraging the easily quantifiable capacity of personnel and equipment. The measure of capacity is articulated through the amount of time—be it in hours, minutes, or seconds—that employees are ready to engage in work tasks. Alternbatively machine time can be used to gauge the capacity more accurately in departments where the pace of work is determined by equipment capabilities. In essence, TDABC emphasizes the pivotal role of time as a measure of capacity, whether attributed to human resources or machinery, in the calculation of costs associated with various activities.


![Time-Driven Activity-Based Costing (TDABC)](/blog/TDABC-Schema.png "Time-Driven Activity-Based Costing (TDABC)")
*TDABC-Schema by Everaert et al.*

#### 1. Determining Departmental Costs

Departmental costs encompass all resources utilized by each department. This includes employee salaries and benefits, supervision salaries and benefits, indirect labor costs along with the associated supervision expenses, equipment and technology depreciation utilized by employees and supervisors, employee and supervisor occupancy costs, and additional costs from company support functions.

#### 2. Determining Practical Capacity


```python
import pandas as pd
import numpy as np

departments = pd.DataFrame(columns=[
'Cost',
'# Employees',
'Practical capacity (mins)',
#'$ / Minute'
])

departments.loc['Purchasing'] = [
525000,
5,
475800]

departments.loc['Warehouse'] = [
520000,
10,
951600]

departments.loc['Shipping'] = [
300000,
6,
570960]

departments.loc['Customer Service'] = [
1800000,
25,
2379000]

departments['Cost per Minute'] = np.round(departments.Cost / departments['Practical capacity (mins)'], 2)
departments_format = departments.style.set_caption('Table 1: Departmental costs, the number of employees, practical capacities in minutes and the capacity cost rates per minute')
departments_format
```




<style type="text/css">
</style>
<table id="T_4320c">
  <caption>Table 1: Departmental costs, the number of employees, practical capacities in minutes and the capacity cost rates per minute</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_4320c_level0_col0" class="col_heading level0 col0" >Cost</th>
      <th id="T_4320c_level0_col1" class="col_heading level0 col1" ># Employees</th>
      <th id="T_4320c_level0_col2" class="col_heading level0 col2" >Practical capacity (mins)</th>
      <th id="T_4320c_level0_col3" class="col_heading level0 col3" >Cost per Minute</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_4320c_level0_row0" class="row_heading level0 row0" >Purchasing</th>
      <td id="T_4320c_row0_col0" class="data row0 col0" >525000</td>
      <td id="T_4320c_row0_col1" class="data row0 col1" >5</td>
      <td id="T_4320c_row0_col2" class="data row0 col2" >475800</td>
      <td id="T_4320c_row0_col3" class="data row0 col3" >1.100000</td>
    </tr>
    <tr>
      <th id="T_4320c_level0_row1" class="row_heading level0 row1" >Warehouse</th>
      <td id="T_4320c_row1_col0" class="data row1 col0" >520000</td>
      <td id="T_4320c_row1_col1" class="data row1 col1" >10</td>
      <td id="T_4320c_row1_col2" class="data row1 col2" >951600</td>
      <td id="T_4320c_row1_col3" class="data row1 col3" >0.550000</td>
    </tr>
    <tr>
      <th id="T_4320c_level0_row2" class="row_heading level0 row2" >Shipping</th>
      <td id="T_4320c_row2_col0" class="data row2 col0" >300000</td>
      <td id="T_4320c_row2_col1" class="data row2 col1" >6</td>
      <td id="T_4320c_row2_col2" class="data row2 col2" >570960</td>
      <td id="T_4320c_row2_col3" class="data row2 col3" >0.530000</td>
    </tr>
    <tr>
      <th id="T_4320c_level0_row3" class="row_heading level0 row3" >Customer Service</th>
      <td id="T_4320c_row3_col0" class="data row3 col0" >1800000</td>
      <td id="T_4320c_row3_col1" class="data row3 col1" >25</td>
      <td id="T_4320c_row3_col2" class="data row3 col2" >2379000</td>
      <td id="T_4320c_row3_col3" class="data row3 col3" >0.760000</td>
    </tr>
  </tbody>
</table>




This example presumes that all employees operate on a five-day workweek, compensated for eight hours each day. Alongside their regular schedule, employees are allocated 30 minutes for lunch, two breaks totaling 30 minutes, and 30 minutes for daily training. Consequently, there is a total of six-and-a-half hours (equivalent to 390 minutes per day) available for assigned tasks. Capacity is quantified in minutes for simplicity, as it is considered more comprehensible than fractional hourly units. Factoring in 10 days of vacation and six paid holidays annually for each employee, their availability for work is calculated at 244 days per year.

Often the practical capacity is derived from a theorethical capacity through a discounting unavailablities due to illness training, break times etc. Such a factor is typically around 80% to 90%.


```python
# Alternative capacity estimation example
workdays_per_month = 20
workdays_per_quarter = 60
workhours_per_day = 8
practical_capacity_factor = 0.85

total_quarterly_capacity_mins = workdays_per_quarter * workhours_per_day * employees * 60
total_quarterly_practical_capacity_mins = total_quarterly_capacity_mins * practical_capacity_factor
```


```python
Purchasing_workload = pd.DataFrame(columns=['Num of Transactions',
'Minutes / Transaction',
'Total Minutes']
#'Cost'
                                  )

Purchasing_workload.loc['Purchase orders'] = [400, 10, 4000]
Purchasing_workload.loc['Line items'] = [5600, 5, 28000]
Purchasing_workload.loc['Vendor contacts'] = [85, 15, 1275]

Purchasing_total_minutes = Purchasing_workload['Total Minutes'].sum()
Purchasing_workload
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Num of Transactions</th>
      <th>Minutes / Transaction</th>
      <th>Total Minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Purchase orders</th>
      <td>400</td>
      <td>10</td>
      <td>4000</td>
    </tr>
    <tr>
      <th>Line items</th>
      <td>5600</td>
      <td>5</td>
      <td>28000</td>
    </tr>
    <tr>
      <th>Vendor contacts</th>
      <td>85</td>
      <td>15</td>
      <td>1275</td>
    </tr>
  </tbody>
</table>
</div>




```python
Department_capacity = pd.DataFrame(columns=['Theoretical capacity', 'Practical capacity', 'Utilized capacity', 'Excess capacity', 'Excess capacity cost'])

Department_capacity.loc['Purchasing'] = [39000, 39000, Purchasing_total_minutes, 39000-Purchasing_total_minutes, np.nan ]
"""Capacity used
(475,800 / 244 X 20 days)
33,275 36,602.50
Capacity available 39,000 42,900.00
Excess capacity 5,725 $6,297.50"""

Department_capacity.loc['Purchasing', 'Excess capacity cost'] = Department_capacity.loc['Purchasing', 'Excess capacity'] * departments.loc['Purchasing', 'Cost per Minute']

Department_capacity
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Theoretical capacity</th>
      <th>Practical capacity</th>
      <th>Utilized capacity</th>
      <th>Excess capacity</th>
      <th>Excess capacity cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Purchasing</th>
      <td>39000.0</td>
      <td>39000.0</td>
      <td>33275.0</td>
      <td>5725.0</td>
      <td>6297.5</td>
    </tr>
  </tbody>
</table>
</div>



#### 3. Identifying the Activities


```python
# building time equations for activities

def Total_purchase_orders_processing_time(time_drivers):
    td = time_drivers
    t = (10 * td.num_purchase_orders# (prepare the basic PO)
    + 15 * td.num_contact_vendor # (if contact vendor)
    + 5 * td.num_po_line_items) #times the number of line items
    return t

def Total_order_entry_time(time_drivers):
    td = time_drivers
    t = (5 * td.num_orders#(enter customer header information)
    + 2 * td.new_customer # (if a new customer)
    + 2 * td.num_line_items # number of line items
    + 10 * td.intern_customer * td.num_orders # (if an international customer)
    + 2 * td.num_orders_special_handling # (if special handling required)
    + 5 * td.num_contact_prod_wareh) # (contact production/warehouse)
    return t  

def Total_pick_pack_order_time(time_drivers):
    td = time_drivers
    t = (3 * td.num_orders #(prepare/print pick-pack)
    + 4 * td.num_line_items # number of line items 
    + 1 * td.num_line_items_special_packaging # times the # of line items w/special packing
    + 3 * td.num_cartons) #times the # of cartons
    return t

def Total_storage_cost(storage_drivers):
    sd = storage_drivers
    daily_rate_pallet = 0.6
    daily_rate_shelve_bin = 0.11
    s = (sd.average_shelve_storage_days * sd.num_li_shelve_stored * sd.average_item_shelve_stored_size *  daily_rate_shelve_bin
         + sd.average_pallet_storage_days * sd.num_li_pallet_stored * sd.average_pallet_stored_size * daily_rate_pallet)
    return s

def Total_order_shipping_time(time_drivers):
    td = time_drivers
    t = (8 * td.num_orders #(prepare basic docs)
    + 5 * td.num_cartons #times the # of cartons
    + 10 * td.num_intern_shipment #(if international shipment)
    + 5 * td.num_rush_order) #(if “rush” order)
    return t

def Summed_total_cost(time_drivers):
    total = Total_purchase_orders_processing_time(time_drivers) * departments.loc['Purchasing', 'Cost per Minute'] + \
            Total_order_entry_time(time_drivers) * departments.loc['Customer Service', 'Cost per Minute'] + \
            Total_pick_pack_order_time(time_drivers) * departments.loc['Warehouse', 'Cost per Minute'] + \
            Total_order_shipping_time(time_drivers) * departments.loc['Shipping', 'Cost per Minute']
    return total
```


```python
Cost_objects = pd.DataFrame(columns=['num_orders',
                                     'num_purchase_orders',
                                     'num_contact_vendor',
                                     'num_po_line_items', 
                                     'num_line_items',
                                     'new_customer', #bool
                                     'intern_customer', #bool
                                     'num_orders_special_handling', 
                                     'num_contact_prod_wareh', 
                                     'num_cartons', 
                                     'num_line_items_special_packaging',
                                     'num_rush_order',
                                     'num_intern_shipment',
                                     
                                    ])
```


```python
Cost_objects.loc['Customer A'] = [2, 0, 0, 0, 200, False, False, 0, 0, 6, 0, 0, 0]
Cost_objects.loc['Customer B'] = [12, 0, 0, 0, 200, False, False, 4, 10, 12, 20, 3, 0]
Cost_objects.loc['Customer C'] = [5, 0, 0, 0, 100, False, True,  0, 0, 5, 5, 5, 5]

costs = pd.DataFrame(columns=['time_only', 'time_and_storage_drivers']).fillna(0)

Cost_objects
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_orders</th>
      <th>num_purchase_orders</th>
      <th>num_contact_vendor</th>
      <th>num_po_line_items</th>
      <th>num_line_items</th>
      <th>new_customer</th>
      <th>intern_customer</th>
      <th>num_orders_special_handling</th>
      <th>num_contact_prod_wareh</th>
      <th>num_cartons</th>
      <th>num_line_items_special_packaging</th>
      <th>num_rush_order</th>
      <th>num_intern_shipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Customer A</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>200</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Customer B</th>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>200</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
      <td>10</td>
      <td>12</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Customer C</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>100</td>
      <td>False</td>
      <td>True</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
costs.loc['Customer A', 'time_only'] = Summed_total_cost(Cost_objects.loc['Customer A'])
costs.loc['Customer B', 'time_only'] = Summed_total_cost(Cost_objects.loc['Customer B'])
costs.loc['Customer C', 'time_only'] = Summed_total_cost(Cost_objects.loc['Customer C'])
```


```python
costs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time_only</th>
      <th>time_and_storage_drivers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Customer A</th>
      <td>789.18</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Customer B</th>
      <td>974.91</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Customer C</th>
      <td>522.45</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The information can be readily expanded to incorporate customer profitability by encompassing the sales of customers A and B, along with their specific sales discount arrangements, product costs, and the resultant product margins. Another crucial aspect influencing account profitability is the assortment of products acquired by A and B. The task for management lies in applying insights derived from the TDABC model outcomes to pinpoint strategies for effectively managing each customer and enhancing their individual profitability.

This result is identical to the example provided by Barndt et al. and rightly so they hinted at incorporating the storage costs into it. Let's do exactly this and apply storage costs into this costing effort. Instead of a continuous space unit we use the number of standardized pallet and shelf-bin sizes as unit.

Other units to measure the capacity could be:
- Vehicle capacity (volume), 
- Warehouse space (Cubic meters)
- Vehicle capacity (Weight: Kilograms)
- Data storage capacity (Gigabytes)
- Data bandwidth (Mbit/s)


### Extending the Time-Driven Activity Based Costing by the storage resource

#### Other resource capacities measured in other units than time minutes are fully compatible


```python
Cost_objects2 = pd.DataFrame(columns=['num_orders',
                                     'num_purchase_orders',
                                     'num_contact_vendor',
                                     'num_po_line_items', 
                                     'num_line_items',
                                     'new_customer', #bool
                                     'intern_customer', #bool
                                     'num_orders_special_handling', 
                                     'num_contact_prod_wareh', 
                                     'num_cartons', 
                                     
                                     'average_days_shelve_stored', # new
                                     'average_size_item_shelve_stored', # new
                                     'num_ordered_items_shelve_stored', # new                                 
                                     'average_days_pallet_stored', # new
                                     'average_size_item_pallet_stored', # new
                                     'num_ordered_items_pallet_stored', # new
                                     
                                     'num_line_items_special_packaging',
                                     'num_rush_order',
                                     'num_intern_shipment'
                                     
                                    ])


Cost_objects2.loc['Customer A'] = [2, 0, 0, 0,  200, False,  False, 0, 0, 6,  40, 0.05, 260, 30, 0.05, 260, 0, 0, 0]
Cost_objects2.loc['Customer B'] = [12, 0, 0, 0, 200, False, False, 4, 10, 12, 20, 0.02, 560, 30, 0.02, 200, 3, 0, 0]
Cost_objects2.loc['Customer C'] = [5, 0, 0, 0, 100, False, True, 0, 0, 5, 12, 0.01, 300, 14, 0.01, 100, 5, 5, 5]

Cost_objects2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_orders</th>
      <th>num_purchase_orders</th>
      <th>num_contact_vendor</th>
      <th>num_po_line_items</th>
      <th>num_line_items</th>
      <th>new_customer</th>
      <th>intern_customer</th>
      <th>num_orders_special_handling</th>
      <th>num_contact_prod_wareh</th>
      <th>num_cartons</th>
      <th>average_days_shelve_stored</th>
      <th>average_size_item_shelve_stored</th>
      <th>num_ordered_items_shelve_stored</th>
      <th>average_days_pallet_stored</th>
      <th>average_size_item_pallet_stored</th>
      <th>num_ordered_items_pallet_stored</th>
      <th>num_line_items_special_packaging</th>
      <th>num_rush_order</th>
      <th>num_intern_shipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Customer A</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>200</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>40</td>
      <td>0.05</td>
      <td>260</td>
      <td>30</td>
      <td>0.05</td>
      <td>260</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Customer B</th>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>200</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
      <td>10</td>
      <td>12</td>
      <td>20</td>
      <td>0.02</td>
      <td>560</td>
      <td>30</td>
      <td>0.02</td>
      <td>200</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Customer C</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>100</td>
      <td>False</td>
      <td>True</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>12</td>
      <td>0.01</td>
      <td>300</td>
      <td>14</td>
      <td>0.01</td>
      <td>100</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
def Total_storage_cost(storage_drivers):
    sd = storage_drivers
    daily_rate_pallet = 0.6
    daily_rate_shelve_bin = 0.11
    s = (sd.average_days_shelve_stored * sd.num_ordered_items_shelve_stored * sd.average_size_item_shelve_stored *  daily_rate_shelve_bin
         + sd.average_days_pallet_stored * sd.num_ordered_items_pallet_stored * sd.average_size_item_pallet_stored * daily_rate_pallet)
    return s

def Summed_total_cost_incl_storage(resource_drivers):
    total = Total_purchase_orders_processing_time(resource_drivers) * departments.loc['Purchasing', 'Cost per Minute'] + \
            Total_order_entry_time(resource_drivers) * departments.loc['Customer Service', 'Cost per Minute'] + \
            Total_pick_pack_order_time(resource_drivers) * departments.loc['Warehouse', 'Cost per Minute'] + \
            Total_storage_cost(resource_drivers) + \
            Total_order_shipping_time(resource_drivers) * departments.loc['Shipping', 'Cost per Minute']
    return total


```


```python
costs.loc['Customer A', 'time_and_storage_drivers'] = Summed_total_cost_incl_storage(Cost_objects2.loc['Customer A'])
costs.loc['Customer B', 'time_and_storage_drivers'] = Summed_total_cost_incl_storage(Cost_objects2.loc['Customer B'])
costs.loc['Customer C', 'time_and_storage_drivers'] = Summed_total_cost_incl_storage(Cost_objects2.loc['Customer C'])

def highlight2(x, color, color2):
    return np.where(x == np.nanmax(x.to_numpy()), f"color: {color};", np.where(x == np.nanmin(x.to_numpy()), f"color: {color2};", None))

costs_format = costs.copy()
#costs_format = costs_format.style.set_caption('Table 1: Departmental costs, the number of employees, practical capacities in minutes and the capacity cost rates per minute')
costs_format.style.apply(highlight2, color='red', color2='green')
```




<style type="text/css">
#T_fa326_row0_col1, #T_fa326_row1_col0 {
  color: red;
}
#T_fa326_row2_col0, #T_fa326_row2_col1 {
  color: green;
}
</style>
<table id="T_fa326">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_fa326_level0_col0" class="col_heading level0 col0" >time_only</th>
      <th id="T_fa326_level0_col1" class="col_heading level0 col1" >time_and_storage_drivers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_fa326_level0_row0" class="row_heading level0 row0" >Customer A</th>
      <td id="T_fa326_row0_col0" class="data row0 col0" >789.180000</td>
      <td id="T_fa326_row0_col1" class="data row0 col1" >1080.380000</td>
    </tr>
    <tr>
      <th id="T_fa326_level0_row1" class="row_heading level0 row1" >Customer B</th>
      <td id="T_fa326_row1_col0" class="data row1 col0" >974.910000</td>
      <td id="T_fa326_row1_col1" class="data row1 col1" >1054.250000</td>
    </tr>
    <tr>
      <th id="T_fa326_level0_row2" class="row_heading level0 row2" >Customer C</th>
      <td id="T_fa326_row2_col0" class="data row2 col0" >522.450000</td>
      <td id="T_fa326_row2_col1" class="data row2 col1" >534.810000</td>
    </tr>
  </tbody>
</table>




### How to implement TDABC into a company?

In the initial **phase of preparation**, a cross-functional team will be assembled with expertise in data analysis and modeling, assigning specific roles and responsibilities. The project's objectives will be clearly defined to address specific business challenges, and the scope will be established, with consideration for a pilot model to assess benefits, costs, and obstacles on a smaller scale.

Moving into Phase II of **data definition**, a comprehensive analysis of data sources will be conducted, gathering pertinent information to drive the initial model. Collaboration with different departments will identify relevant activities, processes, and associated key data points. Time equations and capacity cost rates will be estimated to provide insights into resource utilization.

In Phase III, the **pilot model** will be developed, establishing links between data and the model while ensuring the seamless distribution of general ledger data to departments. Time equations will be loaded into the modeling framework, validated for accuracy, and transaction data integrated for real-time updates. The model will then be employed to calculate costs or profitability, with results validated against real-world scenarios. Thorough validation checks will be conducted to ensure the accuracy and reliability of the model. 

A key objective in developing the pilot model is to acquire the knowledge necessary for automating the integration of data feeds from ERP and other systems into software explicitly designed for conducting TDABC calculations and generating management reports.

By following this phased approach, the implementation team aims to build a robust model aligned with organizational objectives, providing valuable insights into costs, profitability, and relevant business processes.

### Exploring the Positive Effects of TDABC Information on Logistics Decision-Making

The application of Time-Driven Activity-Based Costing (TDABC) significantly enhances logistics decision-making by providing valuable insights and improving various facets of operational efficiency and profitability management. TDABC enables a comprehensive ***understanding of the profitability of logistics service strategies***, offering the management team detailed profitability reports for individual customers. 
Such analysis can reveal that a mere 20 percent of customers contribute to 200 percent of actual profits, while 30 percent of customers eliminate the entirety of actual profits. This prompts strategic initiatives such as renegotiating contracts, focusing sales efforts on profitable customers, and optimizing transportation routes and/or services offered.

Furthermore, TDABC promotes ***higher efficiency in the logistics process*** by furnishing reliable cost information, allowing for the measurement of logistics performance against key drivers. It facilitates detailed monthly cost calculations, aiding in supplier profitability assessment and capacity utilization analysis for enhanced purchasing efficiency. The model's use of time equations enables a nuanced evaluation of productivity, taking into account the type of order and service level provided to customers.

TDABC serves as a catalyst for ***improved profitability management***, guiding the implementation of initiatives such as contract renegotiation and the introduction of policies to optimize delivery routes and minimize costs. It also aids in capacity planning by identifying underused and overused capacity, leading to organizational changes for a better balance of resources.

Furthermore, TDABC ***supports internal benchmarking*** by comparing time equations across different warehouses, fostering the adoption of best practices for increased efficiency. The model ensures ***higher involvement of sales and warehouse managers in profitability management*** by providing specific, manageable data for cost and profitability analysis, fostering a greater awareness of profits and costs among operational employees and sales teams. The integration of financial reporting at the management level with operational cost and profitability reporting enhances overall decision-making and organizational effectiveness.

### References:

Barndt, R. J., Oehlers, P. F., & Soltis, G. S. (2015). Time-Driven Activity-Based Costing: A Powerful Cost Model. *Today's CPA*, March/April, 24-29.

Everaert, Patricia & Bruggeman, Werner & Sarens, Gerrit & Anderson, Steven & Levant, Yves. (2008). Cost Modeling in Logistics Using Time-driven ABC: Experiences from a Wholesaler. *International Journal of Physical Distribution & Logistics Management*, 38, 172-191. DOI: [10.1108/09600030810866977](https://doi.org/10.1108/09600030810866977)


Kaplan, R. S. & Anderson, S. R. (2007). Time-driven activity-based costing: a simpler and more powerful path to higher profits. Boston: Harvard Business School Press.




```python

```

