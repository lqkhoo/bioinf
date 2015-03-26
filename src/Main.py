"""

UCL Bioinformatics 2015 coursework

Standalone protein cellular location classifier based solely on amino acid sequences


Author   : Li Quan Khoo
Date     : 2015

Depencencies:
   Requires : Python 2.7 32-bit (32-bit is a numpy requirement)
              Numpy
              BioPython 1.65
              Scipy
              Scikit-learn

Figure generation requires a basic Matlab installation to run the script,
  but it is not required for classification

"""
from __future__ import division
import os
from Bio import SeqIO
from Bio.Data import IUPACData
from Bio.SeqUtils.ProtParam import ProteinAnalysis, ProtParamData
from sklearn import linear_model, cross_validation
from nt import lstat


DEBUG = 1
DATA_FILES_DIR_PATH = "../data/"		# training data dir path. The file name is the label
MATLAB_FILES_DIR_PATH = "../matlab/data/"
BLIND_SET_LABEL = "blind"			# label for the unlabelled test set


# Files for labels here will not be loaded. Useful for testing smaller sets of data
IGNORE_LABELS = []
# IGNORE_LABELS = ['cyto', 'nucleus']
# IGNORE_LABELS = ['mito', 'secreted']

# Set of individual labels except blind test set and all those specified in IGNORE_LABELS
labels = []

# These fields are set up together during the process of augmenting BioPython data with training data parameters
amino_acid_counts = {}
amino_acid_proportions = {}
expected_X_residue_molecular_weight = 0.0
expected_B_residue_molecular_weight = 0.0
expected_J_residue_molecular_weight = 0.0
expected_Z_residue_molecular_weight = 0.0
max_protein_molecular_weight = 0.0
max_protein_sequence_length = 0.0

# training set of protein sequences
training_sequences = {}
# test set
blind_sequences = []

# training set as feature vectors
training_vectors = {}
# test set
blind_vectors = []

# Classifier models
models = {}

class FeatureVector:
	
	# Feature-specific fields
	min_by_label = {}
	max_by_label = {}
	num_of_vectors_by_label = {}
	sum_by_label = {}
	sum_of_squares_by_label = {}
	
	# Global fields
	
	min_global = {}
	max_global = {}
	num_of_vectors_global = {}
	sum_global = {}
	sum_of_squares_global = {}
	
	# Returns keys to all features in each vector. Requires at least one vector to have been initialized first, otherwise returns empty list
	@classmethod
	def getGlobalFeatures(cls):
		return list(cls.min_global.keys())
	
	# Use this method to assign a key-value pair to a FeatureVector and update global values at the same time.
	# Assign to self._dict directly if global values shouldn't be updated, such as during normalization
	def _set(self, key, value):
		
		feature = key
		
		# Assign value to dict
		self._dict[feature] = value
		
		# Do not update global values when creating blind / test set vectors
		if(feature != BLIND_SET_LABEL):
			
			label = self.label
			
			# Null field checks
			if(not self.__class__.min_by_label.has_key(label)):
				self.__class__.min_by_label[label] = {}
			if(not self.__class__.min_by_label[label].has_key(feature)):
				self.__class__.min_by_label[label][feature] = float("inf")
				
			if(not self.__class__.max_by_label.has_key(label)):
				self.__class__.max_by_label[label] = {}
			if(not self.__class__.max_by_label[label].has_key(feature)):
				self.__class__.max_by_label[label][feature] = 0.0
			
			if(not self.__class__.num_of_vectors_by_label.has_key(label)):
				self.__class__.num_of_vectors_by_label[label] = {}
			if(not self.__class__.num_of_vectors_by_label[label].has_key(feature)):
				self.__class__.num_of_vectors_by_label[label][feature] = 0
			
			if(not self.__class__.sum_by_label.has_key(label)):
				self.__class__.sum_by_label[label] = {}
			if(not self.__class__.sum_by_label[label].has_key(feature)):
				self.__class__.sum_by_label[label][feature] = 0.0
				
			if(not self.__class__.sum_of_squares_by_label.has_key(feature)):
				self.__class__.sum_of_squares_by_label[label] = {}
			if(not self.__class__.sum_of_squares_by_label[label].has_key(feature)):
				self.__class__.sum_of_squares_by_label[label][feature] = 0.0
			
			if(not self.__class__.min_global.has_key(feature)):
				self.__class__.min_global[feature] = float("inf")
			if(not self.__class__.max_global.has_key(feature)):
				self.__class__.max_global[feature] = 0.0
			if(not self.__class__.num_of_vectors_global.has_key(feature)):
				self.__class__.num_of_vectors_global[feature] = 0
			if(not self.__class__.sum_global.has_key(feature)):
				self.__class__.sum_global[feature] = 0.0
			if(not self.__class__.sum_of_squares_global.has_key(feature)):
				self.__class__.sum_of_squares_global[feature] = 0.0
			
			
			# Update global values
			self.__class__.min_by_label[label][feature] = min(value, self.__class__.min_by_label[label][feature])
			self.__class__.max_by_label[label][feature] = max(value, self.__class__.max_by_label[label][feature])
			self.__class__.sum_by_label[label][feature] += value
			self.__class__.sum_of_squares_by_label[label][feature] += value * value
			
			self.__class__.min_global[feature] = min(value, self.__class__.min_global[feature])
			self.__class__.max_global[feature] = max(value, self.__class__.max_global[feature])
			self.__class__.sum_global[feature] += value
			self.__class__.sum_of_squares_global[feature] += value * value
	
	
	def __init__(self, seqRecord, label):
		
		# other fields
		self._pa = ProteinAnalysis(str(seqRecord.seq))
		self._array_representation = None
		
		# bookkeeping fields
		self._seqRecord = seqRecord
		self.label = label
		# features
		
		self._dict = {}
		
		self._set('seq_len', len(seqRecord.seq))
		self._set('isoelectric_point', self._pa.isoelectric_point())
		self._set('aromaticity', self._pa.aromaticity())
		
		_flexibility = self._pa.flexibility()
		self._set('flexibility_global', sum(_flexibility) / len(_flexibility))
		
		"""
		self._dict['seq_len'] = len(seqRecord.seq) / max_protein_sequence_length # normalized to 0 and ~1
		self._dict['isoelectric_point'] = self._pa.isoelectric_point() / 14 # divide by pH 14 to normalize to between 0 and 1
		self._dict['aromaticity'] = self._pa.aromaticity() # already normalized to between 0 and 1 as it's a proportion
		# self._dict['instability_index'] = self._pa.instability_index()
		
		_flexibility = self._pa.flexibility();
		self._dict['flexibility_mean'] = sum(_flexibility) / len(_flexibility); # already normalized to between 0 and 1
		"""
		
		_x_residues_count = str(seqRecord.seq).count('X')
		_b_residues_count = str(seqRecord.seq).count('B')
		_j_residues_count = str(seqRecord.seq).count('J')
		_z_residues_count = str(seqRecord.seq).count('Z')
		_cleaned_seq = str(seqRecord.seq).replace('X', "").replace('B', "").replace('J', "").replace('Z', "")
		_mol_weight_pa = ProteinAnalysis(_cleaned_seq)
		_mol_weight = _mol_weight_pa.molecular_weight()
		_mol_weight += _x_residues_count * expected_X_residue_molecular_weight
		_mol_weight += _b_residues_count * expected_B_residue_molecular_weight
		_mol_weight += _j_residues_count * expected_J_residue_molecular_weight
		_mol_weight += _z_residues_count * expected_Z_residue_molecular_weight
		
		self._set('molecular_weight', _mol_weight)
		self._set('molecular_weight_mean', _mol_weight / len(seqRecord.seq))
		self._set('hydrophobicity_global', self._pa.protein_scale(ProtParamData.kd, len(seqRecord.seq))[0])
		
		"""
		self._dict['molecular_weight_relative'] = _mol_weight / max_protein_molecular_weight # normalize to between 0 and 1
		self._dict['molecular_weight_mean'] = _mol_weight / len(seqRecord.seq) / IUPACData.protein_weights['O'] # normalize to between 0 and 1
		self._dict['hydrophobicity_global'] = (self._pa.protein_scale(ProtParamData.kd, len(seqRecord.seq))[0] + 4.5) / 4.5 # normalize to between 0 and 1
		"""
		
		# other potential features
		
		# Hydrophobicity regions with window sizes 6 and 20
		# http://web.expasy.org/protscale/protscale-doc.html
		
		# Emini surface accessibility (probably not be useful for our application)
		# http://tools.immuneepitope.org/tools/bcell/tutorial.jsp
		
		# Specific sequences
		# Hash substrings and their counts with sliding window over first, last 50 amino acids for each cluster
		
		_helix_fraction, _turn_fraction, _sheet_fraction = self._pa.secondary_structure_fraction() # proportion which is between 0 and 1
		self._set('structure_helix_fraction', _helix_fraction)
		self._set('structure_turn_fraction', _turn_fraction)
		self._set('structure_sheet_fraction', _sheet_fraction)
		
		"""
		self._dict['structure_helix_fraction'], \
		self._dict['structure_turn_fraction'], \
		self._dict['structure_sheet_fraction'] = self._pa.secondary_structure_fraction() # proportion which is between 0 and 1
		"""
		
		_percentages = self._pa.get_amino_acids_percent()
		for _key in _percentages.keys():
			self._set(''.join(['amino_', _key, '_fraction']), _percentages[_key])
		
		"""
		_percentages = self._pa.get_amino_acids_percent()
		for _key in _percentages.keys():
			self._dict['amino_' + _key + '_fraction'] = _percentages[_key] # / amino_acid_proportions[_key] # normalize to full range between 0 and 1 to not under-represent feature
		"""
		
	
	def __getitem__(self, key):
		return self._dict.__getitem__(key)
	
	def __str__(self):
		return str(self._dict)
	
	# This normalizes all features to between 0 and 1 using global min/max values
	# This ensures that no feature dominates due to having a greater range of values
	# Global values shouldn't be updated
	def normalizeAllFeatures(self):
		keys = self.toSortedKeys()
		
		for _key in keys:
			_value = self._dict[_key]
			_min = self.__class__.min_by_label[self.label][_key]
			_max = self.__class__.max_by_label[self.label][_key]
			_new_value = (_value - _min) / (_max - _min)
			self._dict[_key] = _new_value
		
	
	# This normalizes the entire vector to a unit vector
	# Global values shouldn't be updated
	def normalizeToUnitVector(self):
		keys = self.toSortedKeys()
		abs_length = 0.0
		for _key in keys:
			abs_length += pow(self._dict[_key], 2)
		
		for _key in keys:
			self._dict[_key] = self._dict[_key] / abs_length

	
	def toSortedKeys(self):
		return sorted(self._dict.keys())
	
	
	# Returns an array of numbers in order of keys in self.dict sorted by ascending alphabetical order
	def toArray(self):
		
		if(self._array_representation == None):
			self._array_representation = []
			for _key in self.toSortedKeys():
				self._array_representation.append(self._dict[_key])
		
		return self._array_representation
	
	def toCsvString(self):
		array_rep = self.toArray()		
		str_rep = ','.join(map(str, array_rep))
		return str_rep


# Load data from file into memory
def loadDataSets():
	
	print("Loading training sets...")
	
	file_paths = map(lambda local_path: DATA_FILES_DIR_PATH + local_path, os.listdir(DATA_FILES_DIR_PATH)) # get each relative file path
	
	for _path in file_paths:
		_label = _path.replace(".fasta", "").replace(DATA_FILES_DIR_PATH, "")
		_seq_records = SeqIO.parse(_path, "fasta")
		
		if(_label in IGNORE_LABELS):
			continue
		
		if(_label != BLIND_SET_LABEL):
			labels.append(_label)
			training_sequences[_label] = []
			training_sequences[_label].extend(_seq_records)
			
		else:
			blind_sequences.extend(_seq_records)
	
	print("   Finished loading training sets.")

def calculateDerivedData():
	
	"""
	B = N or D
	J = I or L
	Z = E or Q
	"""
	
	print("Calculating training-data-specific parameters...")
	
	# Aggregate all training data
	all_training_sequences = []
	for _key in training_sequences.keys():
		all_training_sequences.extend(training_sequences[_key])
	
	# Calculate background distribution of amino acid residues in training data
	counts = ProteinAnalysis("").count_amino_acids()
	
	additional_keys = ['B', 'Z', 'J', 'U', 'O', 'X']
	for _key in additional_keys:
		counts[_key] = 0
	
	for _seqRecord in all_training_sequences:
		
		_seq_str = str(_seqRecord.seq)
		_pa = ProteinAnalysis(_seq_str)
		
		_am20_counts = _pa.count_amino_acids()
		for _key in _am20_counts.keys():
			counts[_key] += _am20_counts[_key]
			
			for additional_key in additional_keys:
				counts[additional_key] += _seq_str.count(additional_key)
	
	# Total number of residues
	total_residues = sum(counts.values())
	
	# Proportions of all amino acid residues with respect to total number of residues
	proportions = {}
	for _key in counts.keys():
		proportions[_key] = counts[_key] / total_residues
	
	# Correctly calculate expected proportions of ambiguous residues
	proportions['B'] = (proportions['N'] + proportions['D']) / 2
	proportions['Z'] = (proportions['E'] + proportions['Q']) / 2
	proportions['J'] = (proportions['I'] + proportions['L']) / 2
	
	# Calculate expected molecular weight of ambiguous residues
	_x_weight = 0.0
	_aas = list("ACDEFGHIKLMNPQRSTVWY") # use only standard amino acids
	for _aa in _aas:
		_x_weight += proportions[_aa] * IUPACData.protein_weights[_aa]
	_x_weight = _x_weight / len(_aas)

	_b_weight = 0.0
	_prop_total = 0.0
	_aas = list("ND")
	for _aa in _aas:
		_prop_total += proportions[_aa]
		_b_weight += proportions[_aa] * IUPACData.protein_weights[_aa]
	_b_weight = _b_weight / _prop_total
	
	_j_weight = 0.0
	_prop_total = 0.0
	_aas = list("EQ")
	for _aa in _aas:
		_prop_total += proportions[_aa]
		_j_weight += proportions[_aa] * IUPACData.protein_weights[_aa]
	_j_weight = _j_weight / _prop_total
	
	_z_weight = 0.0
	_prop_total = 0.0
	_aas = list("IL")
	for _aa in _aas:
		_prop_total += proportions[_aa]
		_z_weight += proportions[_aa] * IUPACData.protein_weights[_aa]
	_z_weight = _z_weight / _prop_total
	
	
	print("   Finished calculations.")
	
	# Expose counts and proportions from training data to global vars
	return counts, \
		proportions, \
		_x_weight, \
		_b_weight, \
		_j_weight, \
		_z_weight
	
	
# Augment BioPython amino acid data with ambiguous / X residues
# I do it by monkey-wrenching, but it works
def augmentProtParamData():
	
	print("Augmenting BioPython data with ambiguous / unknown amino acid residues based on training data...")
	
	proportions = amino_acid_proportions
	
	# hydrophobicity
	for _key in ['U', 'O', 'X']:
		ProtParamData.kd[_key] = sum(ProtParamData.kd.values()) / 20
	ProtParamData.kd['B'] = proportions['N'] * ProtParamData.kd['N'] + proportions['D'] * ProtParamData.kd['D']
	ProtParamData.kd['Z'] = proportions['E'] * ProtParamData.kd['E'] + proportions['Q'] * ProtParamData.kd['Q']
	ProtParamData.kd['J'] = proportions['I'] * ProtParamData.kd['I'] + proportions['L'] * ProtParamData.kd['L']
	
	# flexibility
	for _key in ['U', 'O', 'X']:
		ProtParamData.Flex[_key] = sum(ProtParamData.Flex.values()) / 20
	ProtParamData.Flex['B'] = proportions['N'] * ProtParamData.Flex['N'] + proportions['D'] * ProtParamData.Flex['D']
	ProtParamData.Flex['Z'] = proportions['E'] * ProtParamData.Flex['E'] + proportions['Q'] * ProtParamData.Flex['Q']
	ProtParamData.Flex['J'] = proportions['I'] * ProtParamData.Flex['I'] + proportions['L'] * ProtParamData.Flex['L']
	
	# hydrophilicity
	for _key in ['U', 'O', 'X']:
		ProtParamData.hw[_key] = sum(ProtParamData.hw.values()) / 20
	ProtParamData.hw['B'] = proportions['N'] * ProtParamData.hw['N'] + proportions['D'] * ProtParamData.hw['D']
	ProtParamData.hw['Z'] = proportions['E'] * ProtParamData.hw['E'] + proportions['Q'] * ProtParamData.hw['Q']
	ProtParamData.hw['J'] = proportions['I'] * ProtParamData.hw['I'] + proportions['L'] * ProtParamData.hw['L']
	
	# Emini surface
	for _key in ['U', 'O', 'X']:
		ProtParamData.em[_key] = sum(ProtParamData.em.values()) / 20
	ProtParamData.em['B'] = proportions['N'] * ProtParamData.em['N'] + proportions['D'] * ProtParamData.em['D']
	ProtParamData.em['Z'] = proportions['E'] * ProtParamData.em['E'] + proportions['Q'] * ProtParamData.em['Q']
	ProtParamData.em['J'] = proportions['I'] * ProtParamData.em['I'] + proportions['L'] * ProtParamData.em['L']
	
	# Janin Interior
	for _key in ['U', 'O', 'X']:
		ProtParamData.ja[_key] = sum(ProtParamData.ja.values()) / 20
	ProtParamData.ja['B'] = proportions['N'] * ProtParamData.ja['N'] + proportions['D'] * ProtParamData.ja['D']
	ProtParamData.ja['Z'] = proportions['E'] * ProtParamData.ja['E'] + proportions['Q'] * ProtParamData.ja['Q']
	ProtParamData.ja['J'] = proportions['I'] * ProtParamData.ja['I'] + proportions['L'] * ProtParamData.ja['L']
	
	print("   Finished.")
# Start of script -------------------------

def generateFeatureVectors():
	
	print("Generating feature vectors...")
	
	# training sets
	for _label in training_sequences.keys():
		training_vectors[_label] = []
		_seq_records = training_sequences.get(_label)
		
		print("   Generating for label \"" + _label + "\"")
		
		for _record in _seq_records:
			_fv = FeatureVector(_record, _label)
			training_vectors[_label].append(_fv)
	
	# blind set
	print("   Generating for label \"" + BLIND_SET_LABEL + "\"")
	for _record in blind_sequences:
		_fv = FeatureVector(_record, BLIND_SET_LABEL)
		blind_vectors.append(_fv)
		
	print("   Finished generating feature vectors.")

def normalizeFeatureVectors(normalize_individual_features = True, normalize_to_unit_vector = False):
	print("Normalizing feature vectors...")
	print("   Normalize individual features = " + str(normalize_individual_features))
	print("   Normalize to unit vector      = " + str(normalize_to_unit_vector))
	
	for _label in training_vectors.keys():
		_fvs = training_vectors[_label]
		for _fv in _fvs:
			if(normalize_individual_features == True):
				_fv.normalizeAllFeatures()
			if(normalize_to_unit_vector == True):
				_fv.normalizeToUnitVector()
	
	for _fv in blind_vectors:
		if(normalize_individual_features == True):
			_fv.normalizeAllFeatures()
		if(normalize_to_unit_vector == True):
			_fv.normalizeToUnitVector()
	
	print("   Finished normalization")


# Writes feature vectors to csv files
def featureVectorsToCsvFiles():
	
	print("Writing vectors to files...")
	f_training = open(MATLAB_FILES_DIR_PATH + 'training.csv', 'w')
	keys_printed = False
	
	for label in training_vectors.keys():
		f_label = open(MATLAB_FILES_DIR_PATH + label + '.csv', 'w')
		for fv in training_vectors[label]:
			if(keys_printed == False):
				_str_out = ',' + ','.join(fv.toSortedKeys()) + '\n'
				f_label.write(_str_out)
				f_training.write(_str_out)
				keys_printed = True
			_str_out = label + "," + fv.toCsvString() + '\n'
			f_label.write(_str_out)
			f_training.write(_str_out)
		f_label.close()
	f_training.close()
	
	# blind set
	keys_printed = False
	f_blind = open(MATLAB_FILES_DIR_PATH + 'test.csv', 'w')
	for fv in blind_vectors:
		if(keys_printed == False):
			_str_out = ',' + ','.join(fv.toSortedKeys()) + '\n'
			f_blind.write(_str_out)
			keys_printed = True
		_str_out = BLIND_SET_LABEL + "," + fv.toCsvString() + '\n'
		f_blind.write(_str_out)
	f_blind.close()
	
	print("   Finished.")


# n must be a float between or equal to 0.0 and 1.0. This is the proportion of items to use for testing
def partition(lst, train_size = 0.5):
	if train_size == 1.0:
		return lst, []
	else:
		return cross_validation.train_test_split(lst, train_size = train_size)

	# Partitions a list into n almost-equal length lists
	# http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
	"""
	q, r = divmod(len(lst), n)
	indices = [q*i + min(i, r) for i in xrange(n+1)]
	return [lst[indices[i]:indices[i+1]] for i in xrange(n)]
	"""

# Trains a linear regressor classifier given a training set
def train(label, training_set):
	
	xs = [] # Arrays
	ys = [] # Labels
	for _label2 in labels: # For each separately labeled vector in the training set,
		for _fv in training_set[_label2]:
			xs.append(_fv.toArray()) # Convert vector to array and make that the x's
			bool_label = 1 if _fv.label == label else 0 # Relabel to either isClass or isNotClass
			ys.append(bool_label)
	
	print("   Training model for label \"" + label + "\".")
	# model = LogisticRegression.train(xs, ys) 
	model = linear_model.LogisticRegression();
	model.fit(xs, ys);
	
	return model

# Validates a model against a validation set
def validate(model, label, validation_set):
	
	print("   Validating...")
	
	xs_v = [] # Validation vectors as arrays
	ys_v = [] # Validation set labels
	for _label2 in labels: # For each separately labeled vector in the training set,
		for _fv in validation_set[_label2]:
			xs_v.append(_fv.toArray()) # Convert vector to array and make that the x's
			bool_label = 1 if _fv.label == label else 0 # Relabel to either isClass or isNotClass
			ys_v.append(bool_label)
	
	true_pos = 0
	false_pos = 0
	true_neg = 0
	false_neg = 0
	for _i in xrange(len(ys_v)):
		_original_label = ys_v[_i]
		# _classifiers_label = LogisticRegression.classify(model, xs_v[_i])
		_classifiers_label = model.predict(xs_v[_i])
		if(_classifiers_label == _original_label):
			if(_original_label == 0): # Original label is negative
				true_neg += 1
			else: # Original label is positive
				true_pos += 1
		else: # classifier label and original label is different
			if(_original_label == 0): # Original label is negative, classifier label is positive
				false_pos += 1
			else: # Original label is positive, classifier label is negative
				false_neg += 1
	
	precision = true_pos / (true_pos + false_pos)
	recall = true_pos / (true_pos + false_neg)
	
	return {
				'true+': true_pos,
				'false+': false_pos,
				'true-': true_neg,
				'false-': false_neg,
				'precision': precision,
				'recall': recall
			}

# Returns possible classifications for a feature vector given a dictionary of trained models
def classify(model_dict, fv):
	label_dict = {}
	for _label in model_dict.keys():
		_model = model_dict[_label]
		_vector = fv.toArray()
		# _classifiers_label = LogisticRegression.classify(model_dict[_label], _vector)
		_classifiers_label = _model.predict(_vector)[0]
		# _error = _model.beta[0] + np.dot(_model.beta[1:], _vector)
		_error = _model.decision_function(_vector)[0]
		label_dict[_label] = (_classifiers_label, _error)
	return label_dict
	
# Trains the classifier models
# Train size is a float between 0 and 1 that declares the proportion of available training data to use for training
def train_models(train_size = 1.0):
	
	print("Training models...")
	# Set up classifier models
	
	training_set = {}
	for _label in labels:
		training_set[_label], _test_set_unused = partition(training_vectors[_label], train_size = train_size)
	
	for _label in labels:
		models[_label] = train(_label, training_set)
	print("   Finished.")
	
# Classifies blind set using traind models
def classify_blind_set():
	
	print("Classifying blind/test set...")
	print("   Results in the form [sequence name]: {binary_classifier_name: (label, error), ... }")
	for _fv in blind_vectors:
		label_dict = classify(models, _fv)
		print("   " + _fv._seqRecord.name + ": " + str(label_dict))
	
	print("   Finished.")
	
	
# Runs cross validation on the training set
def cross_validate(k_folds = 10):
	
	print("Cross-validating...")
	print("   Folds = " + str(k_folds))
	
	means = {}
	
	# Partition training set into training and validation sets
	# training_vectors_partitioned = {}
	
	"""
	for _label in labels:
		training_vectors_partitioned[_label] = partition(training_vectors[_label], k_folds)
	"""
	
	training_vectors_training = {}
	training_vectors_validation = {}
	
	models = {}
	
	# Set up models
	for _label in labels:
		models[_label] = []
	
	# Main loop
	for _fold in xrange(k_folds):
		
		# This is not strictly k-folds. It generates a random partition each time
		for _label in labels:
			training_vectors_training[_label], training_vectors_validation[_label] = partition(training_vectors[_label], 1.0/k_folds)
			# training_vectors_validation[_label], training_vectors_training[_label] = partition(training_vectors[_label], 1.0/k_folds)
		
		"""
		# Set up validation and training sets
		for _label in labels:
			training_vectors_validation[_label] = training_vectors_partitioned[_label][_fold]
			training_vectors_training[_label] = []
			
		for _i in xrange(k_folds):
			if _i == _fold:
				continue # Ignore the fold used for validation
			for _label in labels:
				training_vectors_training[_label].extend(training_vectors_partitioned[_label][_i])
		"""
		
		# For each class that we have to classify,
		for _label in labels:
			
			model = train(_label, training_vectors_training)
			validation_result = validate(model, _label, training_vectors_validation)
			
			if not means.has_key(_label):
				means[_label] = {}
			
			for _metric in validation_result.keys():
				if not means[_label].has_key(_metric):
					means[_label][_metric] = 0.0
				means[_label][_metric] += validation_result[_metric]
			
			models[_label].append((model, validation_result))
			print("      Classifier performance for label \"" + _label + "\" (fold " + str(_fold) + "): " + str(validation_result))
	
	for _label in labels:
		for _metric in means[_label].keys():
			means[_label][_metric] = means[_label][_metric] / k_folds;
		
		print("   Mean classifier performance for label \"" + _label + "\": " + str(means[_label]))

loadDataSets()

amino_acid_counts, \
amino_acid_proportions, \
expected_X_residue_molecular_weight, \
expected_B_residue_molecular_weight, \
expected_J_residue_molecular_weight, \
expected_Z_residue_molecular_weight = calculateDerivedData()

augmentProtParamData()

# do stuff after this line to ensure data has been correctly loaded ---------

generateFeatureVectors()
normalizeFeatureVectors(normalize_individual_features=True, normalize_to_unit_vector=False)
featureVectorsToCsvFiles()
cross_validate(k_folds = 100)

train_models(train_size = 1.0)
classify_blind_set()
