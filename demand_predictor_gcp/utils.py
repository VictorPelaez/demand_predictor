import pandas as pd
import numpy as np
import urllib
import json
import datetime


# get data function and plot series

def get_daily_demand_data(start_date, end_date, write_json=False):
  """
    Description:
      Gets daily demand dataset from REE (spanish)
      link: https://www.ree.es/es/apidatos#request
    Input: 
      - start_date: format 'Y-m-d' as '2019-06-01'
      - end_date: 
    Output:
      - values:
      - times:
  """

  link = 'https://apidatos.ree.es/en/datos/demanda/evolucion?start_date='+start_date+'T00:00&end_date='+end_date+'T22:00&time_trunc=day'
  with urllib.request.urlopen(link) as url:
    s_raw = url.read()
    s = json.loads(s_raw)
    # print(s)

    if write_json:
      with open('data.json', 'w') as fp:
        json.dump(s, fp)

    values_dict = s['included'][0]['attributes']['values']
    values = [item['value'] for item in values_dict]
    times = [item['datetime'].split('T')[0] for item in values_dict]
    times = [datetime.datetime.strptime(t,"%Y-%m-%d").date() for t in times]
    return  np.array(values, dtype="float32"), np.array(times)