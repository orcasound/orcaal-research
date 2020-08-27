import os
import librosa
import numpy as np
import pandas as pd


import os

def find_files(path, substr, return_path=True, search_subdirs=False, search_path=False):
    """ Find all files in the specified directory containing the specified substring in their file name

        Args:
            path: str
                Directory path
            substr: str
                Substring contained in file name
            return_path: bool
                If True, path to each file, relative to the top directory. 
                If false, only return the filenames 
            search_subdirs: bool
                If True, search all subdirectories
            search_path: bool
                Search for substring occurrence in relative path rather than just the filename

        Returns:
            files: list (str)
                Alphabetically sorted list of file names

        Examples:
            >>> from ketos.data_handling.data_handling import find_files
            >>>
            >>> # Find files that contain 'super' in the name;
            >>> # Do not return the relative path
            >>> find_files(path="ketos/tests/assets", substr="super", return_path=False)
            ['super_short_1.wav', 'super_short_2.wav']
            >>>
            >>> # find all files with '.h5" in the name
            >>> # Return the relative path
            >>> find_files(path="ketos/tests/", substr="super", search_subdirs=True)
            ['assets/super_short_1.wav', 'assets/super_short_2.wav']
    """
    # find all files
    all_files = []
    if search_subdirs:
        for dirpath, _, files in os.walk(path):
            if return_path:
                all_files += [os.path.relpath(os.path.join(dirpath, f), path) for f in files]
            else:
                all_files += files
    else:
        all_files = os.listdir(path)

    # select those that contain specified substring
    if isinstance(substr, str): substr = [substr]
    files = []
    for f in all_files:
        for ss in substr:
            if search_path: s = f
            else: s = os.path.basename(f)
            if ss in s:
                files.append(f)
                break

    # sort alphabetically
    files.sort()
    return files


def find_wave_files(path, return_path=True, search_subdirs=False, search_path=False):
    """ Find all wave files in the specified directory

        Args:
            path: str
                Directory path
            return_path: bool
                If True, path to each file, relative to the top directory. 
                If false, only return the filenames 
            search_subdirs: bool
                If True, search all subdirectories
            search_path: bool
                Search for substring occurrence in relative path rather than just the filename

        Returns:
            : list (str)
                Alphabetically sorted list of file names

        Examples:
            >>> from ketos.data_handling.data_handling import find_wave_files
            >>>
            >>> find_wave_files(path="ketos/tests/assets", return_path=False)
            ['2min.wav', 'empty.wav', 'grunt1.wav', 'super_short_1.wav', 'super_short_2.wav']

    """
    return find_files(path, substr=['.wav', '.WAV'], 
        return_path=return_path, search_subdirs=search_subdirs, search_path=search_path)

def str_is_int(s, signed=True):
    """ Check if a given string represents a (signed) integer.

        Args:
            s: str
                Input string.
            signed: bool
                Check if string represents a signed integer (default) or unsigned.

        Returns:
            res: bool
                Result of check
    """
    if signed:
        res = s.isdigit() or (s.startswith('-') and s[1:].isdigit()) or (s.startswith('+') and s[1:].isdigit())
    else:
        res = s.isdigit()
    return res


def unfold(table, sep=','):
    """ Unfolds rows containing multiple labels.

        Args:
            table: pandas DataFrame
                Annotation table.
            sep: str
                Character used to separate multiple labels.

        Returns:
            : pandas DataFrame
                Unfolded table
    """
    df = table
    df = df.astype({'label': 'str'})
    s = df.label.str.split(",").apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = 'label'
    del df['label']
    df = df.join(s)
    return df

def rename_columns(table, mapper):
    """ Renames the table headings to conform with the ketos naming convention.

        Args:
            table: pandas DataFrame
                Annotation table.
            mapper: dict
                Dictionary mapping the headings of the input table to the 
                standard ketos headings.

        Returns:
            : pandas DataFrame
                Table with new headings
    """
    return table.rename(columns=mapper)

def standardize(table=None, filename=None, sep=',', mapper=None, signal_labels=None,\
    backgr_labels=[], unfold_labels=False, label_sep=',', trim_table=False,
    return_label_dict=False):
    """ Standardize the annotation table format.

        The input table can be passed as a pandas DataFrame or as the filename of a csv file.
        The table may have either a single label per row, in which case unfold_labels should be set 
        to False, or multiple labels per row (e.g. as a comma-separated list of values), in which 
        case unfold_labels should be set to True and label_sep should be specified.

        The table headings are renamed to conform with the ketos standard naming convention, following the 
        name mapping specified by the user. 

        Signal labels are mapped to integers 1,2,3,... while background labels are mapped to 0, 
        and any remaining labels are mapped to -1.

        Note that the standardized output table has two levels of indices, the first index being the 
        filename and the second index the annotation identifier. 

        Args:
            table: pandas DataFrame
                Annotation table.
            filename: str
                Full path to csv file containing the annotation table. 
            sep: str
                Separator. Only relevant if filename is specified. Default is ",".
            mapper: dict
                Dictionary mapping the headings of the input table to the 
                standard ketos headings.
            signal_labels: list, or list of lists
                Labels of interest. Will be mapped to 1,2,3,...
                Several labels can be mapped to the same integer by using nested lists. For example, 
                signal_labels=[A,[B,C]] would result in A being mapped to 1 and B and C both being mapped 
                to 2.
            backgr_labels: list
                Labels will be grouped into a common "background" class (0).
            unfold_labels: bool
                Should be set to True if any of the rows have multiple labels. 
                Shoudl be set to False otherwise (default).
            label_sep: str
                Character used to separate multiple labels. Only relevant if unfold_labels is set to True. Default is ",".
            trim_table: bool
                Keep only the columns prescribed by the Ketos annotation format.
            return_label_dict: bool
                Return label dictionary. Default is False.

        Returns:
            table_std: pandas DataFrame
                Standardized annotation table
            label_dict: dict
                Dictionary mapping new labels to old labels. Only returned if return_label_dict is True.
    """
    assert table is not None or filename is not None, 'Either table or filename must be specified'

    # load input table
    if filename is None:
        df = table  
    else:
        assert os.path.exists(filename), 'Could not find input file: {0}'.format(filename)
        df = pd.read_csv(filename, sep=sep)

    # rename columns
    if mapper is not None:
        df = df.rename(columns=mapper)

    # if user has provided duration instead of end time, compute end time
    if 'start' in df.columns.values and 'duration' in df.columns.values and 'end' not in df.columns.values:
        df['end'] = df['start'] + df['duration']

    # keep only relevant columns
    if trim_table:
        df = trim(df)

    # check that dataframe has minimum required columns
    mis = missing_columns(df)
    assert len(mis) == 0, 'Column(s) {0} missing from input table'.format(mis)

    if unfold_labels:
        df = unfold(df, sep=label_sep)

    # cast label column to str
    df = df.astype({'label': 'str'})
    # create list of unique labels in input table
    labels = np.sort(np.unique(df['label'].values)).tolist()

    if signal_labels is None:
        signal_labels = [x for x in labels if x not in backgr_labels]

    # cast to str
    backgr_labels = cast_to_str(backgr_labels)
    signal_labels, signal_labels_flat = cast_to_str(signal_labels, nested=True)

    # separate out background labels, if any
    for x in backgr_labels:
        assert x in labels, 'label {0} not found in input table'.format(x)
    
    # discard remaining labels
    discard_labels = [x for x in labels if x not in signal_labels_flat and x not in backgr_labels]

    # create label dictionary and apply to label column in DataFrame
    _label_dict = create_label_dict(signal_labels, backgr_labels, discard_labels)
    df['label'] = df['label'].apply(lambda x: _label_dict.get(x))

    # cast integer dict keys from str back to int
    label_dict = dict()
    for key, value in _label_dict.items():
        if str_is_int(key): key = int(key)
        label_dict[key] = value

    # transform to multi-indexing
    df = use_multi_indexing(df, 'annot_id')

    table_std = df

    if return_label_dict: return table_std, label_dict
    else: return table_std

def use_multi_indexing(df, level_1_name):
    """ Change from single-level indexing to double-level indexing. 
        
        The first index level is the filename while the second 
        index level is a cumulative integer.

        Args:
            table: pandas DataFrame
                Singly-indexed table. Must contain a column named 'filename'. 

        Returns:
            table: pandas DataFrame
                Multi-indexed table.
    """
    df = df.set_index([df.filename, df.index])
    df = df.drop(['filename'], axis=1)
    df = df.sort_index()
    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values(0), df.groupby(level=0).cumcount()],
        names=['filename', level_1_name])

    return df

def trim(table):
    """ Keep only the columns prescribed by the Ketos annotation format.

        Args:
            table: pandas DataFrame
                Annotation table. 

        Returns:
            table: pandas DataFrame
                Annotation table, after removal of columns.
    """
    keep_cols = ['filename', 'label', 'start', 'end', 'freq_min', 'freq_max']
    drop_cols = [x for x in table.columns.values if x not in keep_cols]
    table = table.drop(drop_cols, axis=1)
    return table

def missing_columns(table, has_time=False):
    """ Check if the table has the minimum required columns.

        Args:
            table: pandas DataFrame
                Annotation table. 
            has_time: bool
                Require time information for each annotation, i.e. start and stop times.

        Returns:
            mis: list
                List of missing columns, if any.
    """
    required_cols = ['filename', 'label']
    if has_time:
        required_cols = required_cols + ['start', 'end']

    mis = [x for x in required_cols if x not in table.columns.values]
    return mis

def is_standardized(table, has_time=False, verbose=True):
    """ Check if the table has the correct indices and the minimum required columns.

        Args:
            table: pandas DataFrame
                Annotation table. 
            has_time: bool
                Require time information for each annotation, i.e. start and stop times.
            verbose: bool
                If True and the table is not standardized, print a message with an example table in the standard format.

        Returns:
            res: bool
                True if the table has the standardized Ketos format. False otherwise.
    """
    required_indices = ['filename', 'annot_id']
    required_cols = ['label']
    if has_time:
        required_cols = required_cols + ['start', 'end']

    mis_cols = [x for x in required_cols if x not in table.columns.values]
    res = (table.index.names == required_indices) and (len(mis_cols) == 0)

    message = """ Your table is not in the Ketos format.

            It should have two levels of indices: filename and annot_id.
            It should also contain at least the 'label' column.
            If your annotations have time information, these should appear in the 'start' and 'end' columns

            extra columns are allowed.

            Here is a minimum example:

                                 label
            filename  annot_id                    
            file1.wav 0          2
                      1          1
                      2          2
            file2.wav 0          2
                      1          2
                      2          1


            And here is a table with time information and a few extra columns ('min_freq', 'max_freq' and 'file_time_stamp')

                                 start   end  label  min_freq  max_freq  file_time_stamp
            filename  annot_id                    
            file1.wav 0           7.0   8.1      2    180.6     294.3    2019-02-24 13:15:00
                      1           8.5  12.5      1    174.2     258.7    2019-02-24 13:15:00
                      2          13.1  14.0      2    183.4     292.3    2019-02-24 13:15:00
            file2.wav 0           2.2   3.1      2    148.8     286.6    2019-02-24 13:30:00
                      1           5.8   6.8      2    156.6     278.3    2019-02-24 13:30:00
                      2           9.0  13.0      1    178.2     304.5    2019-02-24 13:30:00

    
    """
    if res == False and verbose == True:
        print(message)            

    return res

def create_label_dict(signal_labels, backgr_labels, discard_labels):
    """ Create label dictionary, following the convetion:

            * signal_labels are mapped to 1,2,3,...
            * backgr_labels are mapped to 0
            * discard_labels are mapped to -1

        Args:
            signal_labels: list, or list of lists
                Labels of interest. Will be mapped to 1,2,3,...
                Several labels can be mapped to the same integer by using nested lists. For example, 
                signal_labels=[A,[B,C]] would result in A being mapped to 1 and B and C both being mapped 
                to 2.
            backgr_labels: list
                Labels will be grouped into a common "background" class (0).
            discard_labels: list
                Labels will be grouped into a common "discard" class (-1).

        Returns:
            label_dict: dict
                Dict that maps old labels to new labels.
    """
    label_dict = dict()    
    for l in discard_labels: label_dict[l] = -1
    for l in backgr_labels: label_dict[l] = 0
    num = 1
    for l in signal_labels:
        if isinstance(l, list):
            for ll in l:
                label_dict[ll] = num

        else:
            label_dict[l] = num

        num += 1

    return label_dict

def label_occurrence(table):
    """ Identify the unique labels occurring in the table and determine how often 
        each label occurs.

        The input table must have the standardized Ketos format, see 
        :func:`data_handling.selection_table.standardize`. In particular, each 
        annotation should have only a single label value.

        Args:
            table: pandas DataFrame
                Input table.

        Results:
            occurrence: dict
                Dictionary where the labels are the keys and the values are the occurrences.
    """
    occurrence = table.groupby('label').size().to_dict()
    return occurrence

def cast_to_str(labels, nested=False):
    """ Convert every label to str format. 

        If nested is set to True, a flattened version of the input 
        list is also returned.

        Args:
            labels: list
                Input labels
            nested: bool
                Indicate if the input list contains (or may contain) sublists.
                False by default. If True, a flattened version of the 
                list is also returned.

        Results:
            labels_str: list
                Labels converted to str format
            labels_str_flat: list
                Flattened list of labels. Only returned if nested is set to True.
    """
    if not nested:
        labels_str = [str(x) for x in labels]
        return labels_str

    else:
        labels_str = []
        labels_str_flat = []
        for x in labels:
            if isinstance(x, list):
                sublist = []
                for xx in x:
                    labels_str_flat.append(str(xx))
                    sublist.append(str(xx))

                labels_str.append(sublist)

            else:
                labels_str_flat.append(str(x))
                labels_str.append(str(x))

        return labels_str, labels_str_flat

def select(annotations, length, step=0, min_overlap=0, center=False,\
    discard_long=False, keep_id=False):
    """ Generate a selection table by defining intervals of fixed length around 
        every annotated section of the audio data. Each selection created in this 
        way is chracterized by a single, integer-valued, label.

        The input table must have the standardized Ketos format and contain call-level 
        annotations, see :func:`data_handling.selection_table.standardize`.

        The output table uses two levels of indexing, the first level being the 
        filename and the second level being a selection id.

        The generated selections have uniform length given by the length argument. 
        
        Note that the selections may have negative start times and/or stop times 
        that exceed the file duration.

        Annotations longer than the specified selection length will be cropped, unless the 
        step is set to a value larger than 0.

        Annotations with label -1 are discarded.

        Args:
            annotations: pandas DataFrame
                Input table with call-level annotations.
            length: float
                Selection length in seconds.
            step: float
                Produce multiple selections for each annotation by shifting the selection 
                window in steps of length step (in seconds) both forward and backward in 
                time. The default value is 0.
            min_overlap: float
                Minimum required overlap between the selection interval and the  
                annotation, expressed as a fraction of the selection length. Only used if step > 0. 
                The requirement is imposed on all annotations (labeled 1,2,3,...) except 
                background annotations (labeled 0) which are always required to have an 
                overlap of 1.0.
            center: bool
                Center annotations. Default is False.
            discard_long: bool
                Discard all annotations longer than the output length. Default is False.
            keep_id: bool
                For each generated selection, include the id of the annotation from which 
                the selection was generated.

        Results:
            table_sel: pandas DataFrame
                Output selection table.

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.selection_table import select, standardize
            >>> 
            >>> #Load and inspect the annotations.
            >>> df = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>>
            >>> #Standardize annotation table format
            >>> df, label_dict = standardize(df, return_label_dict=True)
            >>> print(df)
                                start   end  label
            filename  annot_id                    
            file1.wav 0           7.0   8.1      2
                      1           8.5  12.5      1
                      2          13.1  14.0      2
            file2.wav 0           2.2   3.1      2
                      1           5.8   6.8      2
                      2           9.0  13.0      1
            >>> 
            >>> #Create a selection table by defining intervals of fixed 
            >>> #length around every annotation.
            >>> #Set the length to 3.0 sec and require a minimum overlap of 
            >>> #0.16*3.0=0.48 sec between selection and annotations.
            >>> #Also, create multiple time-shifted versions of the same selection
            >>> #using a step size of 1.0 sec.     
            >>> df_sel = select(df, length=3.0, step=1.0, min_overlap=0.16, center=True, keep_id=True) 
            >>> print(df_sel.round(2))
                              label  start    end  annot_id
            filename  sel_id                               
            file1.wav 0           2   5.05   8.05         0
                      1           1   6.00   9.00         1
                      2           2   6.05   9.05         0
                      3           1   7.00  10.00         1
                      4           2   7.05  10.05         0
                      5           1   8.00  11.00         1
                      6           1   9.00  12.00         1
                      7           1  10.00  13.00         1
                      8           1  11.00  14.00         1
                      9           2  11.05  14.05         2
                      10          1  12.00  15.00         1
                      11          2  12.05  15.05         2
                      12          2  13.05  16.05         2
            file2.wav 0           2   0.15   3.15         0
                      1           2   1.15   4.15         0
                      2           2   2.15   5.15         0
                      3           2   3.80   6.80         1
                      4           2   4.80   7.80         1
                      5           2   5.80   8.80         1
                      6           1   6.50   9.50         2
                      7           1   7.50  10.50         2
                      8           1   8.50  11.50         2
                      9           1   9.50  12.50         2
                      10          1  10.50  13.50         2
                      11          1  11.50  14.50         2
                      12          1  12.50  15.50         2
    """
    df = annotations.copy()
    df['annot_id'] = df.index.get_level_values(1)

    # check that input table has expected format
    assert is_standardized(df, has_time=True), 'Annotation table appears not to have the expected structure.'

    # discard annotations with label -1
    df = df[df['label'] != -1]

    # number of annotations
    N = len(df)

    # compute length of every annotation
    df['length'] = df['end'] - df['start']

    # discard annotations longer than the requested length
    if discard_long:
        df = df[df['length'] <= length]

    # alignment of new annotations relative to original ones
    if center:
        df['start_new'] = df['start'] + 0.5 * (df['length'] - length)
    else:
        df['start_new'] = df['start'] + np.random.random_sample(N) * (df['length'] - length)

    # create multiple time-shited instances of every annotation
    if step > 0:
        df_new = None
        for idx,row in df.iterrows():
            t = row['start_new']

            if row['label'] == 0:
                ovl = 1
            else:
                ovl = min_overlap

            df_shift = time_shift(annot=row, time_ref=t, length=length, min_overlap=ovl, step=step)
            df_shift['filename'] = idx[0]

            if df_new is None:
                df_new = df_shift
            else:
                df_new = pd.concat([df_new, df_shift])

        # sort by filename and offset
        df = df_new.sort_values(by=['filename','start_new'], axis=0, ascending=[True,True]).reset_index(drop=True)

        # transform to multi-indexing
        df = use_multi_indexing(df, 'sel_id')

    # rename index
    df.index.rename('sel_id', level=1, inplace=True) 

    # drop old/temporary columns, and rename others
    df = df.drop(['start', 'end', 'length'], axis=1)
    df = df.rename(columns={"start_new": "start"})
    df['end'] = df['start'] + length

    # keep annotation id
    if not keep_id:
        df = df.drop(columns=['annot_id'])
    else:
        # re-order columns so annot_it appears last
        cols = df.columns.values.tolist()
        p = cols.index('annot_id')
        cols_new = cols[:p] + cols[p+1:] + ['annot_id']
        df = df[cols_new]
        df = df.astype({'annot_id': int}) #ensure annot_id is int

    # ensure label is integer
    df = df.astype({'label':int})

    table_sel = df
    return table_sel

def time_shift(annot, time_ref, length, step, min_overlap):
    """ Create multiple instances of the same selection by stepping in time, both 
        forward and backward.

        The time-shifted instances are returned in a pandas DataFrame with the same columns as the 
        input annotation, plus a column named 'start_new' containing the start times 
        of the shifted instances.

        Args:
            annot: pandas Series or dict
                Reference annotation. Must contain the labels/keys 'start' and 'end'.
            time_ref: float
                Reference time used as starting point for the stepping.
            length: float
                Output annotation length in seconds.
            step: float
                Produce multiple instances of the same selection by shifting the annotation 
                window in steps of length step (in seconds) both forward and backward in 
                time. The default value is 0.
            min_overlap: float
                Minimum required overlap between the selection intervals and the original 
                annotation, expressed as a fraction of the selection length.   

        Results:
            df: pandas DataFrame
                Output annotation table. The start times of the time-shifted annotations are 
                stored in the column 'start_new'.

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.selection_table import time_shift
            >>> 
            >>> #Create a single 2-s long annotation
            >>> annot = {'filename':'file1.wav', 'label':1, 'start':12.0, 'end':14.0}
            >>>
            >>> #Step across this annotation with a step size of 0.2 s, creating 1-s long annotations that 
            >>> #overlap by at least 50% with the original 2-s annotation 
            >>> df = time_shift(annot, time_ref=13.0, length=1.0, step=0.2, min_overlap=0.5)
            >>> print(df.round(2))
                filename  label  start   end  start_new
            0  file1.wav      1   12.0  14.0       11.6
            1  file1.wav      1   12.0  14.0       11.8
            2  file1.wav      1   12.0  14.0       12.0
            3  file1.wav      1   12.0  14.0       12.2
            4  file1.wav      1   12.0  14.0       12.4
            5  file1.wav      1   12.0  14.0       12.6
            6  file1.wav      1   12.0  14.0       12.8
            7  file1.wav      1   12.0  14.0       13.0
            8  file1.wav      1   12.0  14.0       13.2
            9  file1.wav      1   12.0  14.0       13.4
    """
    if isinstance(annot, dict):
        row = pd.Series(annot)
    elif isinstance(annot, pd.Series):
        row = annot.copy()
    
    row['start_new'] = np.nan
    
    t = time_ref
    t1 = row['start']
    t2 = row['end']

    t_min = t1 - (1 - min_overlap) * length
    t_max = t2 - min_overlap * length

    num_steps_back = 0
    num_steps_forw = 0

    if t_min < t: num_steps_back = int(np.floor((t - t_min) / step))
    if t_max > t: num_steps_forw = int(np.floor((t_max - t) / step))

    row['start_new'] = time_ref
    rows_new = [row]

    # step backwards
    for i in range(num_steps_back):
        ri = row.copy()
        ri['start_new'] = t - (i + 1) * step
        rows_new.append(ri)

    # step forwards
    for i in range(num_steps_forw):
        ri = row.copy()
        ri['start_new'] = t + (i + 1) * step
        rows_new.append(ri)

    # create DataFrame
    df = pd.DataFrame(rows_new)

    # sort according to new start time
    df = df.sort_values(by=['start_new'], axis=0, ascending=[True]).reset_index(drop=True)

    return df

def file_duration_table(path, search_subdirs=False):
    """ Create file duration table.

        Args:
            path: str
                Path to folder with audio files (*.wav)
            search_subdirs: bool
                If True, search include also any audio files in subdirectories.
                Default is False.

        Returns:
            df: pandas DataFrame
                File duration table. Columns: filename, duration
    """
    paths = find_wave_files(path=path, return_path=True, search_subdirs=search_subdirs)
    durations = [librosa.get_duration(filename=os.path.join(path,p)) for p in paths]
    return pd.DataFrame({'filename':paths, 'duration':durations})

def create_rndm_backgr_selections(annotations, files, length, num, no_overlap=False, trim_table=False):
    """ Create background selections of uniform length, randomly distributed across the 
        data set and not overlapping with any annotations, including those labelled 0.

        The random sampling is performed without regard to already created background 
        selections. Therefore, it is in principle possible that some of the created 
        selections will overlap, although in practice this will only occur with very 
        small probability, unless the number of requested selections (num) is very 
        large and/or the (annotation-free part of) the data set is small in size.

        To avoid any overlap, set the 'no_overlap' to True, but note that this can 
        lead to longer execution times.

        Args:
            annotations: pandas DataFrame
                Annotation table.
            files: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.
            length: float
                Selection length in seconds.
            num: int
                Number of selections to be created.
            no_overlap: bool
                If True, randomly selected segments will have no overlap.
            trim_table: bool
                Keep only the columns prescribed by the Ketos annotation format.

        Returns:
            table_backgr: pandas DataFrame
                Output selection table.

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from ketos.data_handling.selection_table import select
            >>> 
            >>> #Ensure reproducible results by fixing the random number generator seed.
            >>> np.random.seed(3)
            >>> 
            >>> #Load and inspect the annotations.
            >>> df = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>> print(df)
                filename  start   end  label
            0  file1.wav    7.0   8.1      1
            1  file1.wav    8.5  12.5      0
            2  file1.wav   13.1  14.0      1
            3  file2.wav    2.2   3.1      1
            4  file2.wav    5.8   6.8      1
            5  file2.wav    9.0  13.0      0
            >>>
            >>> #Standardize annotation table format
            >>> df, label_dict = standardize(df, return_label_dict=True)
            >>> print(df)
                                start   end  label
            filename  annot_id                    
            file1.wav 0           7.0   8.1      2
                      1           8.5  12.5      1
                      2          13.1  14.0      2
            file2.wav 0           2.2   3.1      2
                      1           5.8   6.8      2
                      2           9.0  13.0      1
            >>>
            >>> #Enter file durations into a pandas DataFrame
            >>> file_dur = pd.DataFrame({'filename':['file1.wav','file2.wav','file3.wav',], 'duration':[18.,20.,15.]})
            >>> 
            >>> #Create randomly sampled background selection with fixed 3.0-s length.
            >>> df_bgr = create_rndm_backgr_selections(df, files=file_dur, length=3.0, num=12, trim_table=True) 
            >>> print(df_bgr.round(2))
                              start    end  label
            filename  sel_id                     
            file1.wav 0        1.06   4.06      0
                      1        1.31   4.31      0
                      2        2.26   5.26      0
            file2.wav 0       13.56  16.56      0
                      1       14.76  17.76      0
                      2       15.50  18.50      0
                      3       16.16  19.16      0
            file3.wav 0        2.33   5.33      0
                      1        7.29  10.29      0
                      2        7.44  10.44      0
                      3        9.20  12.20      0
                      4       10.94  13.94      0
    """
    # compute lengths, and discard segments shorter than requested length
    c = files[['filename','duration']]

    if 'offset' in files.columns.names: c['offset'] = files['offset']
    else: c['offset'] = 0

    c.reset_index(drop=True, inplace=True)
    c['length'] = c['duration'] - length
    c = c[c['length'] >= 0]

    # cumulative length 
    cs = c['length'].cumsum().values.astype(float)
    cs = np.concatenate(([0],cs))

    # output
    filename, start, end = [], [], []

    # randomply sample
    df = pd.DataFrame()
    while (len(df) < num):
        times = np.random.random_sample(num) * cs[-1]
        for t in times:
            idx = np.argmax(t < cs) - 1
            row = c.iloc[idx]
            fname = row['filename']
            start = t - cs[idx] + row['offset']
            end   = start + length

            q = query(annotations, filename=fname, start=start, end=end)
            if len(q) > 0: continue

            if no_overlap and len(df) > 0:
                q = query(df.set_index(df.filename), filename=fname, start=start, end=end)
                if len(q) > 0: continue

            x = {'start':start, 'end':end}
            y = files[files['filename']==fname].iloc[0].to_dict()
            z = {**x, **y}
            df = df.append(z, ignore_index=True)

            if len(df) == num: break

    # sort by filename and offset
    df = df.sort_values(by=['filename','start'], axis=0, ascending=[True,True]).reset_index(drop=True)

    # re-order columns
    col_names = ['filename','start','end']
    if not trim_table:
        names = df.columns.values.tolist()
        for name in col_names: names.remove(name)
        col_names += names

    df = df[col_names]

    df['label'] = 0 #add label

    # transform to multi-indexing
    df = use_multi_indexing(df, 'sel_id')

    return df

def select_by_segmenting(files, length, annotations=None, step=None,
    discard_empty=False, pad=True):
    """ Generate a selection table by stepping across the audio files, using a fixed 
        step size (step) and fixed selection window size (length). 
        
        Unlike the :func:`data_handling.selection_table.select` method, selections 
        created by this method are not characterized by a single, integer-valued 
        label, but rather a list of annotations (which can have any length, including zero).

        Therefore, the method returns not one, but two tables: A selection table indexed by 
        filename and segment id, and an annotation table indexed by filename, segment id, 
        and annotation id.

        Args:
            files: pandas DataFrame
                Table with file durations in seconds. 
                Should contain columns named 'filename' and 'duration'.
            length: float
                Selection length in seconds.
            annotations: pandas DataFrame
                Annotation table.
            step: float
                Selection step size in seconds. If None, the step size is set 
                equal to the selection length.
            discard_empty: bool
                If True, only selection that contain annotations will be used. 
                If False (default), all selections are used.
            pad: bool
                If True (default), the last selection window is allowed to extend 
                beyond the endpoint of the audio file.

        Returns:
            sel: pandas DataFrame
                Selection table
            annot: pandas DataFrame
                Annotations table. Only returned if annotations is specified.

        Example:
            >>> import pandas as pd
            >>> from ketos.data_handling.selection_table import select_by_segmenting, standardize
            >>> 
            >>> #Load and inspect the annotations.
            >>> annot = pd.read_csv("ketos/tests/assets/annot_001.csv")
            >>>
            >>> #Standardize annotation table format
            >>> annot, label_dict = standardize(annot, return_label_dict=True)
            >>> print(annot)
                                start   end  label
            filename  annot_id                    
            file1.wav 0           7.0   8.1      2
                      1           8.5  12.5      1
                      2          13.1  14.0      2
            file2.wav 0           2.2   3.1      2
                      1           5.8   6.8      2
                      2           9.0  13.0      1
            >>>
            >>> #Create file table
            >>> files = pd.DataFrame({'filename':['file1.wav', 'file2.wav', 'file3.wav'], 'duration':[11.0, 19.2, 15.1]})
            >>> print(files)
                filename  duration
            0  file1.wav      11.0
            1  file2.wav      19.2
            2  file3.wav      15.1
            >>>
            >>> #Create a selection table by splitting the audio data into segments of 
            >>> #uniform length. The length is set to 10.0 sec and the step size to 5.0 sec.
            >>> sel = select_by_segmenting(files=files, length=10.0, annotations=annot, step=5.0) 
            >>> #Inspect the selection table
            >>> print(sel[0].round(2))
                              start   end
            filename  sel_id             
            file1.wav 0         0.0  10.0
                      1         5.0  15.0
            file2.wav 0         0.0  10.0
                      1         5.0  15.0
                      2        10.0  20.0
            file3.wav 0         0.0  10.0
                      1         5.0  15.0
                      2        10.0  20.0
            >>> #Inspect the annotations
            >>> print(sel[1].round(2))
                                       start   end  label
            filename  sel_id annot_id                    
            file1.wav 0      0           7.0   8.1      2
                             1           8.5  10.0      1
                      1      0           2.0   3.1      2
                             1           3.5   7.5      1
                             2           8.1   9.0      2
                      2      1           0.0   2.5      1
                             2           3.1   4.0      2
            file2.wav 0      0           2.2   3.1      2
                             1           5.8   6.8      2
                             2           9.0  10.0      1
                      1      1           0.8   1.8      2
                             2           4.0   8.0      1
                      2      2           0.0   3.0      1
    """
    if step is None:
        step = length

    # check that the annotation table has expected format
    if annotations is not None:
        assert is_standardized(annotations, has_time=True), 'Annotation table appears not to have the expected structure.'
        
        annotations = annotations[annotations.label != -1] #discard annotations with label -1

    # create selections table by segmenting
    sel = segment_files(files, length=length, step=step, pad=pad)

    # max number of segments
    num_segs = sel.index.get_level_values(1).max() + 1

    # create annotation table by segmenting
    if annotations is not None:
        annot = segment_annotations(annotations, num=num_segs, length=length, step=step)

        # discard empties
        if discard_empty:
            indices = list(set([(a, b) for a, b, c in annot.index.tolist()]))
            sel = sel.loc[indices].sort_index()

        return sel, annot

    else:
        return sel

def segment_files(table, length, step=None, pad=True):
    """ Generate a selection table by stepping across the audio files, using a fixed 
        step size (step) and fixed selection window size (length). 

        Args:
            table: pandas DataFrame
                File duration table.
            length: float
                Selection length in seconds.
            step: float
                Selection step size in seconds. If None, the step size is set 
                equal to the selection length.
            pad: bool
                If True (default), the last selection window is allowed to extend 
                beyond the endpoint of the audio file.

        Returns:
            df: pandas DataFrame
                Selection table
    """
    if step is None:
        step = length

    # compute number of segments for each file
    table['num'] = (table['duration'] - length) / step + 1
    if pad: 
        table.num = table.num.apply(np.ceil).astype(int)
    else:
        table.num = table.num.apply(np.floor).astype(int)

    df = table.loc[table.index.repeat(table.num)]
    df.set_index(keys=['filename'], inplace=True, append=True)
    df = df.swaplevel()
    df = df.sort_index()
    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values(0), df.groupby(level=0).cumcount()],
        names=['filename', 'sel_id'])

    df['start'] = df.index.get_level_values(1) * step
    df['end'] = df['start'] + length
    df.drop(columns=['num','duration'], inplace=True)

    return df

def segment_annotations(table, num, length, step=None):
    """ Generate a segmented annotation table by stepping across the audio files, using a fixed 
        step size (step) and fixed selection window size (length). 
        
        Args:
            table: pandas DataFrame
                Annotation table.
            num: int
                Number of segments
            length: float
                Selection length in seconds.
            step: float
                Selection step size in seconds. If None, the step size is set 
                equal to the selection length.

        Returns:
            df: pandas DataFrame
                Annotations table
    """
    if step is None:
        step = length

    segs = []
    for n in range(num):
        # select annotations that overlap with segment
        t1 = n * step
        t2 = t1 + length
        a = table[(table.start < t2) & (table.end > t1)].copy()
        if len(a) > 0:
            # shift and crop annotations
            a['start'] = a['start'].apply(lambda x: max(0, x - t1))
            a['end'] = a['end'].apply(lambda x: min(length, x - t1))
            a['sel_id'] = n #map to segment
            segs.append(a)

    df = pd.concat(segs)
    df.set_index(keys=['sel_id'], inplace=True, append=True)
    df = df.swaplevel()
    df = df.sort_index()
    return df

def query(selections, annotations=None, filename=None, label=None, start=None, end=None):
    """ Query selection table for selections from certain audio files 
        and/or with certain labels.

        Args:
            selections: pandas DataFrame
                Selections table
            annotations: pandas DataFrame
                Annotations table. Optional.
            filename: str or list(str)
                Filename(s)
            label: int or list(int)
                Label(s)
            start: float
                Earliest end time in seconds
            end: float
                Latest start time in seconds

        Returns:
            : pandas DataFrame or tuple(pandas DataFrame, pandas DataFrame)
            Selection table, accompanied by an annotation table if an input 
            annotation table is provided.
    """
    if annotations is None:
        return query_labeled(selections, filename, label, start, end)
    else:
        return query_annotated(selections, annotations, filename, label, start, end)

def query_labeled(table, filename=None, label=None, start=None, end=None):
    """ Query selection table for selections from certain audio files 
        and/or with certain labels.

        Args:
            selections: pandas DataFrame
                Selections table, which must have a 'label' column.
            filename: str or list(str)
                Filename(s)
            label: int or list(int)
                Label(s)
            start: float
                Earliest end time in seconds
            end: float
                Latest start time in seconds

        Returns:
            df: pandas DataFrame
            Selection table
    """
    df = table
    if filename is not None:
        if isinstance(filename, str):
            if filename not in df.index: return df.iloc[0:0]
            else: filename = [filename]
        
        df = df.loc[filename]

    if label is not None:
        if not isinstance(label, list):
            label = [label]

        df = df[df.label.isin(label)]

    if start is not None:
        df = df[df.end > start]

    if end is not None:
        df = df[df.start < end]

    return df

def query_annotated(selections, annotations, filename=None, label=None, start=None, end=None):
    """ Query selection table for selections from certain audio files 
        and/or with certain labels.

        Args:
            selections: pandas DataFrame
                Selections table.
            annotations: pandas DataFrame
                Annotations table.
            filename: str or list(str)
                Filename(s)
            label: int or list(int)
                Label(s)
            start: float
                Earliest end time in seconds
            end: float
                Latest start time in seconds

        Returns:
            df1,df2: tuple(pandas DataFrame, pandas DataFrame)
                Selection table and annotation table
    """
    df1 = selections
    df2 = annotations

    df1 = query_labeled(df1, filename=filename, start=start, end=end)
    df2 = query_labeled(df2, filename=filename, label=label, start=start, end=end)

    indices = list(set([x[:-1] for x in df2.index.tolist()]))
    df1 = df1.loc[indices].sort_index()

    return df1, df2