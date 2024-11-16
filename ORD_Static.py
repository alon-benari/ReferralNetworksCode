import pyodbc
import pandas as pd
import json
import re
import os
from collections import namedtuple

class Static(object):
	
	def __init__(self,db ):
		#,driver='ODBC Driver 17 for SQL server',server = 'VhaCdwDwhSql33.vha.med.va.gov', database='D05_TeamDev':
		#try:
		#	q_string = 'DRIVER={'+driver+'};SERVER='+server+';DATABASE='+database+';Trusted_connection=yes;Integrated Security=SSP'
		#	self.conn=pyodbc.connect(q_string)
			
		#except(RuntimeError):
		#	print(RuntimeError)
		self.vet_count = {}
		
		if db =='OIT19':
			self.con = self.OIT19()
		if db =='ORD':
			self.con = self.ORD()
		if db == 'SQL33':
			self.con = self.SQL33

		self.db = db
		
		# self.sta3n_counties = self.load_county_visn() #self.county_visn()
		# self.vet_count_vssc = self.format_count_vets_vssc()

	def OIT19 (self):
			return pyodbc.connect(
				'''
				DRIVER={SQL Server};
				SERVER=OITMACSQL019.va.gov;
				USER=MACClinApps;
				PWD=;
				''',
				autocommit=True
				)

	def SQL33(self):
		return pyodbc.connect(
			'''
			DRIVER={SQL server};
			SERVER=VhaCdwDwhSql33.vha.med.va.gov;
			DATABASE = D05_TeamDev'
			Trusted_connection=yes;Integrated Security=SSP

			''',
			autocommit=True
			)

	def ORD(self):
		return pyodbc.connect(
				'''
				DRIVER={SQL Server};
				SERVER=;
				DATABASE = ;
				Trusted_connection=yes;
				Integrated Security=SSP'
				''',
				autocommit=True
				)
	

	def SQLsubmit(self,sql):
		'''
		A generic version of passing a sql string
		'''
		try:
			data = pd.read_sql(sql=sql, con = self.con)
		
		except(RuntimeError):
			print(RuntimeError)
		
		# self.con.close()
		return data

	def add_formatting(self, string):
		'''
		A method to add quotation marks to a string
		string - a string to be formatted
		'''
		return("'"+string+"'")


	def get_sql(self,sta3n):
		'''
		return a sql string
		'''
		sql_dict ={
			'SQL33':'select  * from [D05_TeamDev].[MAC_CI].[Request130_Main] where Sta3n = {}'.format(sta3n),
			'ORD':'select  * from [ORD_BenAri_202001023D].[Dflt].[Main] where Sta3n = {}'.format(sta3n)
			
		
		}

		return sql_dict[self.db]
		

	def get_data(self, sta3n, start_date = '2019-01-01', end_date = '2019-12-31'):
		'''
		A method to make a get request to OIT019 database
		'''
	
		sql = self.get_sql(sta3n)																													
		try:
			data = pd.read_sql(sql=sql, con = self.con)
			#self.con.close()
			return data
		except(RuntimeError):
			return (RuntimeError)
		
		

	def save_edgelist(self,data):
		'''
		A method to save the recieved data 
		the fname  = 'edge_list'_sta3n_start_date_end_date
		'''
		fname = 'edge_list'+str(self.sta3n)+'_'+self.start_date+'_'+self.end_date
		data.to_csv('D:\\ClinApp\\Development\\VHAMACBENARA\\ORD_ConsultNetwork\\ORD_ConsultNetwork/Assets/'+fname + '.csv')

	
	def get_sta3n(self, data, sta3n):
			'''
			return a dataframe for the a said sta3n 
			as a dictionary
			sta3n:string, station id

			'''
			sta3n_df = data.query("station_id =={}".format(sta3n))
			d = self.format_county_vets()
			station = sta3n.strip("'")
			population = 0
			try:
				population = sum([self.format_count_vets_vssc().loc[str(f)]['Enrollees Priority 1 to 8D']  for f in sta3n_df.fips.tolist()]) #sum([d.loc[f]['year2020'] for f in sta3n_df.fips.tolist()])
				return {
				station:{
				#'fips':sta3n_df.fips.tolist(),
				'population':float(population),
				#'station_name': sta3n_df.station_name.unique()[0],
				#'visn_id':list(str(sta3n_df.visn_id.unique())),
				#'state':sta3n_df.state.unique()[0],
				#'total_rvu':sta3n_df.totalrvu.to_list(),
				#'admparent_fcdmd':sta3n_df.admparent_fcdmd.unique()[0],
				#'admparent_portion':sta3n_df.admparentportion.to_list()
					}
				}
			except:
				print('unable to compute population size:{}'.format(sta3n))
				return {
				station:{
				#'fips':sta3n_df.fips.tolist(),
				'population':float(population),
				#'station_name': sta3n_df.station_name.unique()[0],
				#'visn_id':list(str(sta3n_df.visn_id.unique())),
				#'state':sta3n_df.state.unique(),
				#'total_rvu':sta3n_df.totalrvu.to_list(),
				#'admparent_fcdmd':sta3n_df.admparent_fcdmd.unique()[0],
				#'admparent_portion':sta3n_df.admparentportion.to_list()
					}
				}
			
			
	def county_visn(self):
		'''
		A method to render a json object of {sta3n:{visn: visn_id, fips:[]}
		'''
		data = pd.read_csv('./Resources/catchments.csv')
		data['fips'] = data['fips'].apply(lambda x:str(x).zfill(5)).apply(lambda x: str(x))
		sta3ns = ["'" + s + "'" for s in data.station_id.unique()]

		return {
			s.strip("'"):self.get_sta3n(data, s) for s in sta3ns
			}

	def load_county_visn(self):
		'''
		load the county visn data from  json
		'''
		json_fn = './Resources/county_visn.json'
		try:
			with open(json_fn) as f:
				data =  json.load(f)
				return data
		except:
				print('An error occured, could not parse load file')
		

	def save_population_sta3n(self, dict_object):
		'''
		A method to save the population in each sta3n
		'''


	def save_county_visn(self, data):
		'''
		Save the county visn into a json object
		data -  dict, the output of method county_visn()
		'''
		json_fn = 'county_visn.json'
		

		try:
			os.chdir('./Resources')
			with open(json_fn,'w') as f:
				json.dump(data,f,  indent =4 )
			os.chdir ('..')
		except Exception as e:
			print('cound not save file.{}'.format(e))

	def get_fips(self, string):
		'''
		A method to get the fips code from a string string
		if no brackets pass
		'''
		try:
			return string.split(")")[0].split('(')[1]
		except:
			pass

	def format_count_vets_vssc(self):
		'''
		A method to return the a data frame of fips and counts of veterans using VSSC data source
		'''
		
		os.chdir('./Resources')
		data = pd.read_csv('Enrollees_Users_09282021.csv', error_bad_lines = False)
		os.chdir ('..')
		data.dropna(inplace=True)
		
		data['fips'] = data['Current County State'].apply(lambda x: self.get_fips(x))

		return  data[['fips','Enrollees Priority 1 to 8D']].groupby('fips').sum()

		


	def format_county_vets(self):
		'''
		A method to format the estimates for veterans in county
		this is coming from https://www.va.gov/vetdata/docs/Demographics/New_Vetpop_Model/9L_VetPop2018_County.xlsx

		'''
		os.chdir('./Resources')
		data = pd.read_csv('VetPop2018_County.csv')
		data.dropna(inplace=True)
		data['fips'] = data.fips.apply(lambda x: int(x))
		os.chdir('..')
		return data.set_index('fips')
	
	def load_sta3ns_complexity(self):
		'''
		return the table of sta3ns and their complexity
		'''
		os.chdir('./Resources')
		complexity = pd.read_csv('FacilityComplexHistory.csv')
		os.chdir('..')
		return complexity
		

	def add_population(self, sta3n):
		'''
		Get the expected population size associated with a sta3n
		sta3n : string, station id
		'''

		d = self.format_county_vets()
		
		return {sta3n:
		  {'population':sum([d.loc[f]['year2020'] for f in self.sta3n_counties[sta3n][sta3n]['fips']])
					  }
		}

	def get_sta3n_complexity(self,complexity):
		'''
		Return a list of sta3n by complexity

		complexity -  string 1a, 1b 1c
		'''
		return {complexity:[row[1] for row in  self.load_sta3ns_complexity().iterrows() if complexity in  row[1].MCGName]}


	def get_complex_dict(self, complexity):
		'''
		return sta3ns of complexity
		complexity -  string 1a 1b 1c
		'''
		return {complexity:[(row[1].Sta5a, row[1].AdmParent) for row in  self.load_sta3ns_complexity().iterrows() if complexity in  row[1].MCGName]}

	def get_vet_counts(self):
		'''
		Read the Enrollees, non enrollees and users
		'''
		df = pd.read_csv('./Static/EnrolleesNonEnrollesUsers.csv')
		parent = df['Parent Facility'].unique()
		

		for p in parent:
			sta3n = p.split(' ')[1].strip('(').strip(')')	
			self.vet_count[sta3n] = self.parse_parent_div(p)
			self.vet_count[sta3n]['vet_count'] = df[df['Parent Facility'] == p]['Enrollees Priority 1 to 8D'].sum()
			
		self.vet_count['612'] = self.vet_count['612A4']


	def parse_parent_div(self,parent):
		return {
				'sta3n':parent.split(' ')[1].strip('(').strip(')'),
				'visn':parent.split(' ')[0].strip('(').strip(')'),
				'name':'_'.join(parent.split(' ')[2:]),
				
				}
	

#s = Static('SQL33')
#data = s.get_data(sta3n = 612)
#
#sql = s.sql_consult(sta3n = 612) # get the  correct SQL string 
#data = s.SQLsubmit(sql)
#s.save_edgelist(data) # save the csv file






