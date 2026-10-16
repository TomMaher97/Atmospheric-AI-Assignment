# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:17:07 2026

@author: tom_m
"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#%%
data_dir = Path(__file__)

df = {f.stem: pd.read_csv(f) for f in data_dir.glob("*.csv")}


#%%

util = 0.8  # fraction of nameplate capacity facilities actually run at
naphtha_yield = 0.075  # fraction of naphtha in crude oil 
naphtha_cons = 2.75  # tonnes of naphtha consumed per tonne of ethylene made
naphtha_price = 757  # current ARA naphtha benchmark in USD/t
int_dist = 3  # max km between a refinery and cracker of an integrated complex
tol = 1e-6  # tonnes below which a balance counts as zero

fac = df['facilities'].set_index('id') # facilities table
units = df['units'] # units table
legs = df['transport_legs'] # transport legs table
org_names = df['organizations'].set_index('id')['name'] # organization names table


def haversine(lat1, lon1, lat2, lon2):
    # Returns the shortest distance between two points on a sphere given their coordinates
    R = 6371  # radius of the Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]) # convert degrees to radians
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_distance(t):
    # Adds the geographical distance between the supplier and cracker of each
    # row of table t
    t['distance_km'] = haversine(fac.latitude[t.supplier].values, fac.longitude[t.supplier].values,
                                 fac.latitude[t.cracker].values, fac.longitude[t.cracker].values)
    t.loc[t.supplier == t.cracker, 'distance_km'] = 0 # zero where the refinery and cracker are the same facility
    return t


#%%
# Step 1: actual throughput (ethylene production) of each cracker
cr_units = units[units.unit_type == 'steam_cracker_naphtha'].dropna(subset=['design_capacity_t_per_year']) # removing crackers with no listed design capacity
crackers = cr_units.groupby('facility_id').design_capacity_t_per_year.sum().to_frame('nameplate_t') # groups by facility_id and sums the design capacity of all units at that facility
crackers['throughput_t'] = crackers.nameplate_t * util # calculates actual throughput 

# Refinery crude capacities, converted to naphtha output in step 2
rf_units = units[units.unit_type == 'crude_distillation'].dropna(subset=['design_capacity_t_per_year']) # does the same for refineries
refineries = rf_units.groupby('facility_id').design_capacity_t_per_year.sum().to_frame('nameplate_t')

print('Number of crackers =', len(crackers))
print('Number of refineries =', len(refineries))

#%%
# Step 4: naphtha-importing ports, their locations and 2025 naphtha imports
pt = df['port_throughput'] # port throughput table
ports = pt[(pt.material == 'naphtha') & (pt.imported_t_2025 > 0)].set_index('facility_id') # keeps ports that imported naphtha in 2025
ports = fac[['name', 'latitude', 'longitude']].join(ports[['imported_t_2025', 'exported_t_2025']], how='inner') # adds each port's name and location

print('Number of naphtha-importing ports =', len(ports))
print(ports.sort_values('imported_t_2025', ascending=False).head(10))

# Every facility that can supply naphtha to a cracker
suppliers = pd.concat([pd.DataFrame({'type': 'refinery'}, index=refineries.index),
                       pd.DataFrame({'type': 'port'}, index=ports.index)]) # combines refineries and ports into one table, labelled by type

#%%
# Step 5: empty output table, one row per cracker and supplying refinery/port
out_cols = ['cracker', 'supplier', 'mode', 'naphtha_t', 'share']
out = pd.DataFrame(columns=out_cols) # empty table with these columns, filled in step 19

#%%
# Step 7 (first part): (parent) organisation of each facility, needed for step 6
owners = df['facility_ownership'].groupby('facility_id').organization_id.apply(lambda o: tuple(sorted(o))) # groups by facility_id and creates a tuple of sorted organization_ids for each facility
jv = owners[owners.apply(len) > 1]  # facilities with more than one direct owner

fac['org'] = jv.apply(lambda o: 'JV_' + '_'.join(o))  # same owners give the same JV
fac['org'] = fac.org.fillna(fac.parent_organization_id) # fills in the parent organisation for facilities that aren't JVs
fac['org'] = fac.org.fillna(owners[owners.apply(len) == 1].apply(lambda o: o[0])) # fills in the direct owner for facilities with only one direct owner and no parent organisation

for o in jv:
    org_names['JV_' + '_'.join(o)] = ' / '.join(org_names.get(i, i) for i in o) + ' (JV)' # adds a readable name for each JV made from its owners' names

print('Number of JV facilities =', len(jv))
print('Facilities with no organisation =', fac.org.isna().sum())

#%%
# Step 6: pipelines from a refinery or naphtha-importing port to a cracker
pipes = legs[(legs['mode'] == 'pipeline') & legs.source_facility_id.isin(suppliers.index)
             & legs.destination_facility_id.isin(crackers.index)] # filters the transport legs to only include pipelines that connect suppliers to crackers
pipes = pipes.rename(columns={'source_facility_id': 'supplier', 'destination_facility_id': 'cracker'}) # renames the columns to match the suppliers and crackers dataframes
pipes = pipes.sort_values('distance_km').drop_duplicates(['supplier', 'cracker'])  # keep the shortest
pipes = pipes[['supplier', 'cracker', 'distance_km']].reset_index(drop=True) # resets the index and keeps only the relevant columns
pipes['supplier_org'] = pipes.supplier.map(fac.org) # maps the supplier facility to its organisation
pipes['cracker_org'] = pipes.cracker.map(fac.org) # maps the cracker facility to its organisation
pipe_pairs = set(zip(pipes.supplier, pipes.cracker)) # creates a set of tuples representing the supplier-cracker pairs connected by pipelines

print('Number of supplier-cracker pipelines =', len(pipes))

#%%
# Steps 7 and 8: organisations owning refineries/ports and crackers, split by
# how many of each they own. Integrated complexes are included
sup_org = pd.DataFrame({'supplier': suppliers.index, 'org': fac.org[suppliers.index].values}).dropna() # each supplier and its organisation, removing those with none (all the ports)
cr_org = pd.DataFrame({'cracker': crackers.index, 'org': fac.org[crackers.index].values}).dropna() # does the same for crackers

org_pairs = sup_org.merge(cr_org, on='org')  # every supplier-cracker pair within an organisation
org_pairs['org_name'] = org_pairs.org.map(org_names) # adds the organisation's name
org_pairs['n_sup'] = org_pairs.org.map(sup_org.groupby('org').size()) # number of refineries/ports the organisation owns
org_pairs['n_cr'] = org_pairs.org.map(cr_org.groupby('org').size()) # number of crackers the organisation owns
org_pairs = add_distance(org_pairs) # adds the geographical distance between the supplier and cracker of each row of org_pairs

own_1_1 = org_pairs[(org_pairs.n_sup == 1) & (org_pairs.n_cr == 1)]  # step 7
own_1_n = org_pairs[(org_pairs.n_sup == 1) & (org_pairs.n_cr > 1)]  # step 8
own_n_1 = org_pairs[(org_pairs.n_sup > 1) & (org_pairs.n_cr == 1)]
own_n_n = org_pairs[(org_pairs.n_sup > 1) & (org_pairs.n_cr > 1)]

print('Organisations with one refinery/port and one cracker =', own_1_1.org.nunique())
print('Organisations with one refinery/port and multiple crackers =', own_1_n.org.nunique())
print('Organisations with multiple refineries/ports and one cracker =', own_n_1.org.nunique())
print('Organisations with multiple refineries/ports and multiple crackers =', own_n_n.org.nunique())

#%%
# Step 9: geographical distance between every cracker and every refinery/port.
# For integrated complexes this is under int_dist km, or zero for the same facility
pairs = pd.MultiIndex.from_product([crackers.index, suppliers.index], names=['cracker', 'supplier']).to_frame(index=False) # every cracker-supplier combination
pairs = add_distance(pairs)

# Companies of each facility: its direct owners and its parent organisation
companies = pd.concat([df['facility_ownership'][['facility_id', 'organization_id']],
                       fac.parent_organization_id.dropna().rename('organization_id').rename_axis('facility_id').reset_index()]) # stacks the direct owners and the parent organisations into one table
companies = companies.groupby('facility_id').organization_id.apply(set) # set of companies for each facility

# Integrated complexes, used in step 12: a facility with both a refinery and a
# cracker unit, or a refinery and cracker sharing a company within int_dist km
integrated = pairs[pairs.supplier.isin(refineries.index) & (pairs.distance_km <= int_dist)].copy() # refinery-cracker pairs within int_dist km of each other, including the same facility
shared = [companies.get(s, set()) & companies.get(c, set()) for s, c in zip(integrated.supplier, integrated.cracker)] # companies the refinery and cracker have in common
integrated['company'] = [', '.join(org_names.get(o, o) for o in sorted(x)) for x in shared] # names of the shared companies
integrated['shared'] = [len(x) > 0 for x in shared] # True if the refinery and cracker share at least one company
integrated = integrated[(integrated.supplier == integrated.cracker) | integrated.shared] # keeps the same facility, or pairs that share a company
integrated = integrated.sort_values('distance_km')[['company', 'supplier', 'cracker', 'distance_km']]
int_pairs = set(zip(integrated.supplier, integrated.cracker)) # set of (refinery, cracker) pairs that are integrated complexes

print('Number of integrated complexes =', len(integrated))
i_print = integrated.copy() # copy with facility names instead of ids, for printing
i_print['supplier'] = i_print.supplier.map(fac.name)
i_print['cracker'] = i_print.cracker.map(fac.name)
print(i_print.round(2).to_string(index=False))

# Rail and barge legs between a refinery/port and a cracker, in either direction, used in step 15
rb = legs[legs['mode'].isin(['rail', 'barge'])] # rail and barge legs
to_cr = rb[rb.source_facility_id.isin(suppliers.index) & rb.destination_facility_id.isin(crackers.index)] # legs running to a cracker
from_cr = rb[rb.source_facility_id.isin(crackers.index) & rb.destination_facility_id.isin(suppliers.index)] # legs running from a cracker
rail_barge = pd.concat([to_cr.rename(columns={'source_facility_id': 'supplier', 'destination_facility_id': 'cracker'}),
                        from_cr.rename(columns={'source_facility_id': 'cracker', 'destination_facility_id': 'supplier'})]) # puts both into supplier and cracker columns
rail_barge = rail_barge.sort_values('distance_km') # shortest first

# Mode of the shortest known leg between each pair, used to label steps 14 and 16
leg_modes = legs.sort_values('distance_km').drop_duplicates(['source_facility_id', 'destination_facility_id']) # keeps the shortest leg between each source and destination
leg_modes = dict(zip(zip(leg_modes.source_facility_id, leg_modes.destination_facility_id), leg_modes['mode'])) # dictionary of (source, destination): mode

#%%
bids = df['feedstock_bids']
bids = bids[bids.feedstock == 'naphtha'].set_index('facility_id') # keeps naphtha bids, i.e. those of the crackers
crackers = crackers.join(bids[['bid_min_usd_per_t', 'bid_max_usd_per_t']]) # adds each cracker's minimum and maximum bid, NaN if it has none


def sourcing_model(naphtha_yield, naphtha_cons):
    # Runs steps 2-3 and 10-19 for a given naphtha yield and consumption.
    # Returns the output table, the allocations, and the crackers and
    # suppliers with their remaining balances and labels
    sup = suppliers.copy() # copies so each run starts from the same tables
    cr = crackers.copy()

    # Step 2: naphtha output of each refinery, and naphtha imported at each port
    sup['naphtha_t'] = pd.concat([refineries.nameplate_t * util * naphtha_yield, ports.imported_t_2025]) # crude capacity x utilisation x yield for refineries, 2025 imports for ports

    # Step 3: naphtha feed requirement of each cracker
    cr['feed_t'] = cr.throughput_t * naphtha_cons # ethylene throughput x naphtha consumed per tonne of ethylene

    # Step 10: market naphtha requirement, i.e. the part of the feed the
    # cracker would still buy from an outside seller at the current price
    bmin, bmax = cr.bid_min_usd_per_t, cr.bid_max_usd_per_t
    frac = np.where(bmax <= naphtha_price, 0, (bmax - naphtha_price) / (bmax - bmin)) # 0 if the max bid is at or below the price, otherwise falls linearly between the bids
    frac = np.where((bmin >= naphtha_price) | bmin.isna(), 1, frac) # 1 if the min bid is at or above the price, or there is no bid
    cr['market_t'] = cr.feed_t * frac # fraction of the feed requirement bought on the market
    cr['no_bid'] = bmin.isna()  # flagged, market requirement assumed to be the full feed

    # Step 11: running balances, updated after every allocation
    supply_rem = sup.naphtha_t.copy() # naphtha left at each refinery/port
    feed_rem = cr.feed_t.copy() # feed requirement left at each cracker
    market_rem = cr.market_t.copy() # market requirement left at each cracker
    alloc = [] # list of allocations made, one per row

    def allocate(s, c, mode):
        # Allocates naphtha from supplier s to cracker c. Sales between
        # different organisations not joined by pipeline are limited by, and
        # taken off, the cracker's market requirement. Integrated complexes
        # count as the same organisation
        same_org = (pd.notna(fac.org[s]) and fac.org[s] == fac.org[c]) or (s, c) in int_pairs # True if the supplier and cracker belong to the same organisation
        market = not same_org and (s, c) not in pipe_pairs # True if the sale is limited by the cracker's market requirement
        amount = min(supply_rem[s], feed_rem[c]) # can't supply more than the supplier has left or the cracker still needs
        if market:
            amount = min(amount, market_rem[c])
        if amount <= tol: # nothing to allocate
            return
        supply_rem[s] -= amount # updates the running balances
        feed_rem[c] -= amount
        if market:
            market_rem[c] -= amount
        alloc.append([c, s, mode, amount, market]) # records the allocation

    # Step 12: refinery side to cracker side of each integrated complex
    for row in integrated.itertuples():
        allocate(row.supplier, row.cracker, 'integrated')

    # Step 13: pipelines, shortest first
    for row in pipes.itertuples():
        allocate(row.supplier, row.cracker, 'pipeline')

    # Step 14: facilities of the same organisation, closest pair first
    for row in org_pairs.sort_values('distance_km').itertuples():
        allocate(row.supplier, row.cracker, leg_modes.get((row.supplier, row.cracker), 'unrouted')) # mode of the shortest known leg, or 'unrouted' if there is none

    # Step 15: rail and barge legs between refineries/ports and crackers, shortest first
    for row in rail_barge.itertuples():
        allocate(row.supplier, row.cracker, row.mode)

    # Step 16: any cracker, closest first
    for row in pairs.sort_values('distance_km').itertuples():
        allocate(row.supplier, row.cracker, leg_modes.get((row.supplier, row.cracker), 'unrouted'))

    # Step 17: label any feed requirement left at each cracker
    cr['received_t'] = cr.feed_t - feed_rem # naphtha received = feed requirement - feed requirement left
    cr['feed_rem_t'] = feed_rem
    cr['market_rem_t'] = market_rem
    short = feed_rem > tol # crackers with some feed requirement left
    cr['status'] = 'Fully supplied'
    cr.loc[short, 'status'] = 'Unallocated' # each label below overwrites the one above where both apply
    cr.loc[short & (market_rem <= tol), 'status'] = 'Reduced throughput'
    cr.loc[short & (cr.received_t <= tol) & (cr.market_t <= tol), 'status'] = 'Idle'

    # Step 18: label any naphtha left at each refinery/port
    sup['remaining_t'] = supply_rem
    sup['status'] = np.where(supply_rem > tol, 'surplus', 'fully allocated') # 'surplus' if any naphtha is left

    # Step 19: fill the step 5 table and calculate each cracker's actual throughput
    alloc = pd.DataFrame(alloc, columns=['cracker', 'supplier', 'mode', 'naphtha_t', 'market']) # converts the list of allocations into a table
    o = alloc.groupby(['cracker', 'supplier'], as_index=False).agg(
        mode=('mode', 'first'), naphtha_t=('naphtha_t', 'sum')) # one row per cracker-supplier pair, adding up the naphtha supplied
    o['share'] = o.naphtha_t / o.groupby('cracker').naphtha_t.transform('sum') # naphtha supplied / total received, as transform gives each row its cracker's total
    cr['actual_throughput_t'] = cr.received_t / naphtha_cons # ethylene made from the naphtha received
    cr['pct_nameplate'] = 100 * cr.actual_throughput_t / cr.nameplate_t # actual throughput as a % of nameplate capacity

    return o[out_cols], alloc, cr, sup # o[out_cols] puts the columns in the step 5 order


#%%
# Steps 2-3 and 10-19 with the base parameters
out, alloc, cr, sup = sourcing_model(naphtha_yield, naphtha_cons)

print('Crackers with no bid on file =', cr.no_bid.sum())
print(cr.status.value_counts())
print(sup.groupby('type').status.value_counts())

#%%
# Step 20: checks
share_sums = out.groupby('cracker').share.sum() # sum of each cracker's shares
print('a) Every supplied cracker\'s shares sum to 1:', np.allclose(share_sums, 1))

supplied = alloc.groupby('supplier').naphtha_t.sum().reindex(sup.index, fill_value=0) # total naphtha supplied by each refinery/port, 0 if it supplied none
print('b) No refinery/port supplies more naphtha than it has:', (supplied <= sup.naphtha_t + tol).all())

market_supplied = alloc[alloc.market].groupby('cracker').naphtha_t.sum().reindex(cr.index, fill_value=0) # naphtha each cracker bought on the market
print('c) No cracker gets more than its market requirement from other organisations\' non-pipeline sources:',
      (market_supplied <= cr.market_t + tol).all())

rf_surplus = sup[sup.type == 'refinery'].remaining_t.sum() # total naphtha left at refineries
port_exports = pt[pt.material == 'naphtha'].exported_t_2025.sum() # total naphtha exported by all ports in 2025
print('Total refinery surplus =', round(rf_surplus), 't')
print('Total port naphtha exports =', round(port_exports), 't')
print('d) Refinery surplus >= port naphtha exports:', rf_surplus >= port_exports)

#%%
# Step 19 output table for every cracker
o_print = out.copy() # copy with facility names instead of ids, for printing
o_print['cracker'] = o_print.cracker.map(fac.name)
o_print['supplier'] = o_print.supplier.map(fac.name)
o_print = o_print.sort_values(['cracker', 'share'], ascending=[True, False]) # grouped by cracker, largest share first
print(o_print.round(3).to_string(index=False))

# Status and actual throughput of every cracker, including idle ones with no rows above
c_print = cr[['status', 'feed_t', 'received_t', 'actual_throughput_t', 'pct_nameplate']].copy()
c_print.index = c_print.index.map(fac.name).rename('cracker') # cracker names instead of ids
print(c_print.sort_index().round(1).to_string())

#%%
# Step 21: sensitivity of each share and each cracker's actual throughput to
# naphtha yield and consumption. Steps 4-9 don't depend on either, so only
# steps 2-3 and 10-19 are rerun
scenarios = {'base': (naphtha_yield, naphtha_cons),
             'yield 0.05': (0.05, naphtha_cons),
             'yield 0.1': (0.1, naphtha_cons),
             'cons 2.5': (naphtha_yield, 2.5),
             'cons 3': (naphtha_yield, 3)}

sens = {}
sens_tp = {}
sens_status = {}
for k, (y, n) in scenarios.items():
    o, a, c, s = sourcing_model(y, n) # runs the model with this scenario's yield and consumption
    sens[k] = o.set_index(['cracker', 'supplier']).share # share of each cracker-supplier pair in this scenario
    sens_tp[k] = c.pct_nameplate # each cracker's actual throughput as a % of nameplate
    sens_status[k] = c.status.value_counts() # number of crackers with each status

sens = pd.DataFrame(sens).fillna(0)  # a missing pair supplies a share of zero
sens['range'] = sens.max(axis=1) - sens.min(axis=1) # difference between the largest and smallest share across scenarios
sens = sens.sort_values('range', ascending=False) # most sensitive shares first

print(pd.DataFrame(sens_status).fillna(0).astype(int))
print('\nShares that change most between scenarios')
s_print = sens.head(15).reset_index()
s_print['cracker'] = s_print.cracker.map(fac.name).str[:30] # facility names, cut to 30 characters to fit the printout
s_print['supplier'] = s_print.supplier.map(fac.name).str[:30]
print(s_print.round(3).to_string(index=False))

print('\nActual throughput as a % of nameplate capacity')
sens_tp = pd.DataFrame(sens_tp)
sens_tp.index = sens_tp.index.map(fac.name).str[:30].rename('cracker') # cracker names instead of ids
print(sens_tp.round(1).to_string())

