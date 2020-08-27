import pytest
import os
import ketos.data_handling.selection_table as st
import pandas as pd
import numpy as np
from io import StringIO


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_trim():
    standard = ['filename','label','start','end','freq_min','freq_max']
    extra = ['A','B','C']
    df = pd.DataFrame(columns=extra)
    df = st.trim(df)
    assert len(df.columns) == 0
    df = pd.DataFrame(columns=standard+extra)
    df = st.trim(df)
    assert sorted(df.columns.values) == sorted(standard)

def test_missing_columns():
    standard = ['filename','label','start','end','freq_min','freq_max']
    df = pd.DataFrame(columns=standard)
    assert len(st.missing_columns(df)) == 0
    df = pd.DataFrame(columns=standard[:-1])
    assert len(st.missing_columns(df)) == 0
    df = pd.DataFrame(columns=standard[1:])
    assert sorted(st.missing_columns(df)) == ['filename']

def test_is_standardized():
    df = pd.DataFrame({'filename':'test.wav','label':[1],'start':[0],'end':[2],'freq_min':[None],'freq_max':[None]})
    df = st.standardize(df)
    assert st.is_standardized(df) == True
    df = pd.DataFrame({'filename':'test.wav','label':[1]})
    df = st.standardize(df)
    assert st.is_standardized(df) == True
    df = pd.DataFrame({'filename':'test.wav','label':[1]})
    assert st.is_standardized(df) == False

def test_create_label_dict():
    l1 = [0, 'gg', -17, 'whale']
    l2 = [-33, 1, 'boat']
    l3 = [999]
    d = st.create_label_dict(l1, l2, l3)
    ans = {-33: 0, 1:0, 'boat': 0, 999: -1, 0: 1, 'gg':2, -17: 3, 'whale': 4}
    assert d == ans

def test_create_label_dict_can_handle_nested_list():
    l1 = [0, 'gg', [-17, 'whale']]
    l2 = [-33, 1, 'boat']
    l3 = [999]
    d = st.create_label_dict(l1, l2, l3)
    ans = {-33: 0, 1:0, 'boat': 0, 999: -1, 0: 1, 'gg':2, -17: 3, 'whale': 3}
    assert d == ans

def test_unfold(annot_table_mult_labels):
    res = st.unfold(annot_table_mult_labels)
    ans = pd.DataFrame({'filename':['f0.wav','f0.wav','f1.wav'], 'label':['1','2','3'], 'start':[0,0,1], 'end':[1,1,2]})
    res = res.reset_index(drop=True)[ans.columns]
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_standardize(annot_table_std):
    res = st.standardize(annot_table_std)
    d = '''filename annot_id label  start  end                   
f0.wav   0             3    0.0  3.3
f0.wav   1             2    3.0  6.3
f1.wav   0             4    1.0  4.3
f1.wav   1             2    4.0  7.3
f2.wav   0             5    2.0  5.3
f2.wav   1             1    5.0  8.3'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_standardize_from_file(annot_table_file):
    res, d = st.standardize(filename=annot_table_file, mapper={'fname': 'filename', 'STOP': 'end'}, 
        signal_labels=[1,'k'], backgr_labels=[-99, 'whale'], return_label_dict=True)
    ans = {-99: 0, 'whale':0, 2: -1, 'zebra': -1, 1: 1, 'k':2}
    assert d == ans
    d = '''filename annot_id label  start  end                   
f0.wav   0             1      0    1
f1.wav   0            -1      1    2
f2.wav   0             2      2    3
f3.wav   0             0      3    4
f4.wav   0             0      4    5
f5.wav   0            -1      5    6'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_standardize_with_nested_list(annot_table_file):
    res, d = st.standardize(filename=annot_table_file, mapper={'fname': 'filename', 'STOP': 'end'}, 
        signal_labels=[[1,'whale'],'k'], backgr_labels=[-99], return_label_dict=True)
    ans = {-99: 0, 2: -1, 'zebra': -1, 1: 1, 'whale':1, 'k':2}
    assert d == ans
    d = '''filename annot_id label  start  end                   
f0.wav   0             1      0    1
f1.wav   0            -1      1    2
f2.wav   0             2      2    3
f3.wav   0             0      3    4
f4.wav   0             1      4    5
f5.wav   0            -1      5    6'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])
    
def test_label_occurrence(annot_table_std):
    df = annot_table_std
    oc = st.label_occurrence(df)
    ans = {-1: 1, 0: 2, 1: 1, 2: 1, 3: 1}
    assert oc == ans

def test_select_center(annot_table_std):
    df = st.standardize(annot_table_std)
    # request length shorter than annotations
    res = st.select(df, length=1, center=True)
    d = '''filename sel_id label  start   end
f0.wav   0           3   1.15  2.15
f0.wav   1           2   4.15  5.15
f1.wav   0           4   2.15  3.15
f1.wav   1           2   5.15  6.15
f2.wav   0           5   3.15  4.15
f2.wav   1           1   6.15  7.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])
    # request length longer than annotations
    res = st.select(df, length=5, center=True)
    d = '''filename sel_id  label  start   end
f0.wav   0           3  -0.85  4.15
f0.wav   1           2   2.15  7.15
f1.wav   0           4   0.15  5.15
f1.wav   1           2   3.15  8.15
f2.wav   0           5   1.15  6.15
f2.wav   1           1   4.15  9.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_select_removes_discarded_annotations(annot_table_std):
    df = annot_table_std
    df = st.standardize(df)
    res = st.select(df, length=1, center=True)
    assert len(res[res.label==-1]) == 0

def test_select_enforces_overlap(annot_table_std):
    np.random.seed(3)
    df = annot_table_std
    df = st.standardize(df)
    # requested length: 5.0 sec
    # all annotations have length: 3.3 sec  (3.3/5.0=0.66)
    length = 5.0
    overlap = 0.5
    df_new = st.select(df, length=length, min_overlap=overlap, keep_id=True)
    t1 = df_new.start.values
    t2 = df_new.end.values
    idx = zip(df_new.index.get_level_values(0), df_new.annot_id)
    df = df.loc[idx]
    t2_orig = df.end.values
    t1_orig = df.start.values
    assert np.all(t2 >= t1_orig + overlap * length)
    assert np.all(t1 <= t2_orig - overlap * length)

def test_select_step(annot_table_std):
    df = annot_table_std
    df = st.standardize(df)
    N = len(df[df['label']!=-1])
    K = len(df[df['label']==0])
    df_new = st.select(df, length=1, center=True, min_overlap=0, step=0.5, keep_id=True)
    M = len(df_new)
    assert M == (N - K) * (2 * int((3.3/2+0.5)/0.5) + 1) + K * (2 * int((3.3/2-0.5)/0.5) + 1)
    df_new = st.select(df, length=1, center=True, min_overlap=0.4, step=0.5)
    M = len(df_new)
    assert M == (N - K) * (2 * int((3.3/2+0.5-0.4)/0.5) + 1) + K * (2 * int((3.3/2-0.5)/0.5) + 1)

def test_time_shift(annot_table_std):
    row = pd.Series({'label':3.00,'start':0.00,'end':3.30,'annot_id':0.00,'length':3.30,'start_new':-0.35})
    res = st.time_shift(annot=row, time_ref=row['start_new'], length=4.0, min_overlap=0.8, step=0.5)
    d = '''label  start  end  annot_id  length  start_new
0    3.0    0.0  3.3       0.0     3.3      -0.35'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0])
    pd.testing.assert_frame_equal(ans, res[ans.columns.values])

def test_select_with_varying_overlap(annot_table_std):
    """ Test that the number of selections increases as the 
        minimum required overlap is reduced"""
    df = st.standardize(annot_table_std)
    # request length shorter than annotations
    num_sel = []
    for min_overlap in np.linspace(1.0, 0.0, 11):
        res = st.select(df, length=1.0, step=0.5, center=True, min_overlap=min_overlap)
        num_sel.append(len(res))
    
    assert np.all(np.diff(num_sel) >= 0)
    # request length longer than annotations
    num_sel = []
    for min_overlap in np.linspace(1.0, 0.0, 11):
        res = st.select(df, length=4.0, step=0.5, center=True, min_overlap=min_overlap)
        num_sel.append(len(res))

    assert np.all(np.diff(num_sel) >= 0)

def test_create_rndm_backgr_selections(annot_table_std, file_duration_table):
    np.random.seed(1)
    df = st.standardize(annot_table_std)
    dur = file_duration_table 
    num = 5
    df_bgr = st.create_rndm_backgr_selections(annotations=df, files=dur, length=2.0, num=num)
    assert len(df_bgr) == num
    # assert selections have uniform length
    assert np.all(df_bgr.end.values - df_bgr.start.values == 2.0)
    # assert all selection have label = 0
    assert np.all(df_bgr.label.values == 0)
    # assert selections do not overlap with any annotations
    for bgr_idx, bgr_sel in df_bgr.iterrows():
        start_bgr = bgr_sel.start
        end_bgr = bgr_sel.end
        fname = bgr_idx[0]
        q = st.query(df, start=start_bgr, end=end_bgr, filename=fname)
        assert len(q) == 0

def test_create_rndm_backgr_keeps_misc_cols(annot_table_std, file_duration_table):
    """ Check that the random background selection creation method keeps 
        any miscellaneous columns"""
    np.random.seed(1)
    df = st.standardize(annot_table_std)
    dur = file_duration_table 
    dur['extra'] = 'testing'
    df_bgr = st.create_rndm_backgr_selections(annotations=df, files=dur, length=2.0, num=5)
    assert np.all(df_bgr['extra'].values == 'testing')
    df_bgr = st.create_rndm_backgr_selections(annotations=df, files=dur, length=2.0, num=5, trim_table=True)
    assert 'extra' not in df_bgr.columns.values.tolist()

def test_create_rndm_backgr_files_missing_duration(annot_table_std, file_duration_table):
    """ Check that the random background selection creation method works even when 
        some of the files are missing from the file duration list"""
    np.random.seed(1)
    df = st.standardize(annot_table_std)
    dur = file_duration_table.drop(0) 
    df_bgr = st.create_rndm_backgr_selections(annotations=df, files=dur, length=2.0, num=11)

def test_create_rndm_backgr_nonzero_offset(annot_table_std, file_duration_table):
    """ Check that the random background selection creation method works when file duration 
        table includes offsets"""
    np.random.seed(1)
    df = st.standardize(annot_table_std)
    file_duration_table['offset'] = [1, 2, 3, 4, 5, 6]
    df_bgr = st.create_rndm_backgr_selections(annotations=df, files=file_duration_table, length=2.0, num=11)

def test_create_rndm_backgr_selections_no_overlap(annot_table_std, file_duration_table):
    """ Check that random selections have no overlap"""
    np.random.seed(1)
    df = st.standardize(annot_table_std)
    dur = file_duration_table 
    num = 30
    df_bgr = st.create_rndm_backgr_selections(annotations=df, files=dur, length=2.0, num=num)
    num_overlap = 0
    for idx,row in df_bgr.iterrows():
        q = st.query(df_bgr, filename=idx[0], start=row['start'], end=row['end'])
        num_overlap += len(q) - 1
    
    assert num_overlap > 0

    df_bgr = st.create_rndm_backgr_selections(annotations=df, files=dur, length=2.0, num=num, no_overlap=True)
    num_overlap = 0
    for idx,row in df_bgr.iterrows():
        q = st.query(df_bgr, filename=idx[0], start=row['start'], end=row['end'])
        num_overlap += len(q) - 1
    
    assert num_overlap == 0

def test_select_by_segmenting(annot_table_std, file_duration_table):
    a = st.standardize(annot_table_std)
    f = file_duration_table
    sel = st.select_by_segmenting(f, length=5.1, annotations=a, step=4.0, discard_empty=True, pad=True)
    # check selection table
    d = '''filename sel_id start  end
f0.wav   0         0.0  5.1
f0.wav   1         4.0  9.1
f1.wav   0         0.0  5.1
f1.wav   1         4.0  9.1
f2.wav   0         0.0  5.1
f2.wav   1         4.0  9.1
f2.wav   2         8.0 13.1'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(ans, sel[0][ans.columns.values])
    # check annotation table
    d = '''filename sel_id annot_id label  start  end
f0.wav   0      0             3         0.0        3.3
f0.wav   0      1             2         3.0        5.1
f0.wav   1      1             2         0.0        2.3
f1.wav   0      0             4         1.0        4.3
f1.wav   0      1             2         4.0        5.1
f1.wav   1      0             4         0.0        0.3
f1.wav   1      1             2         0.0        3.3
f2.wav   0      0             5         2.0        5.1
f2.wav   0      1             1         5.0        5.1
f2.wav   1      0             5         0.0        1.3
f2.wav   1      1             1         1.0        4.3
f2.wav   2      1             1         0.0        0.3'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1,2])
    pd.testing.assert_frame_equal(ans, sel[1][ans.columns.values])

def test_query_labeled(annot_table_std):
    df = st.standardize(annot_table_std)
    df = st.select(df, length=1, center=True)
    # query for file that does not exist
    q = st.query_labeled(df, filename='fff.wav')
    assert len(q) == 0
    # query for 1 file
    q = st.query_labeled(df, filename='f1.wav')
    d = '''filename sel_id label  start   end                   
f1.wav  0           4   2.15  3.15
f1.wav  1           2   5.15  6.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(q, ans[q.columns.values])
    # query for 1 file, and 1 that does not exist
    q = st.query_labeled(df, filename=['f1.wav','fff.wav'])
    d = '''filename sel_id label  start   end                   
f1.wav   0           4   2.15  3.15
f1.wav   1           2   5.15  6.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(q, ans[q.columns.values])
    # query for 2 files
    q = st.query_labeled(df, filename=['f1.wav','f2.wav'])
    d = '''filename sel_id label  start   end                            
f1.wav   0           4   2.15  3.15
f1.wav   1           2   5.15  6.15
f2.wav   0           5   3.15  4.15
f2.wav   1           1   6.15  7.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(q, ans[q.columns.values])
    # query for labels
    q = st.query_labeled(df, label=[2,5])
    d = '''filename sel_id label  start   end                   
f0.wav   1           2   4.15  5.15
f1.wav   1           2   5.15  6.15
f2.wav   0           5   3.15  4.15'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(q, ans[q.columns.values])
    # query for label that does not exist
    q = st.query_labeled(df, label=99)
    assert len(q) == 0

def test_query_annotated(annot_table_std, file_duration_table):
    a = st.standardize(annot_table_std)
    f = file_duration_table
    sel = st.select_by_segmenting(f, length=5.1, annotations=a, step=4.0, discard_empty=True, pad=True)
    # query for 1 file
    q1, q2 = st.query_annotated(sel[0], sel[1], label=[2,4])
    d = '''filename sel_id start  end
f0.wav   0         0.0  5.1
f0.wav   1         4.0  9.1
f1.wav   0         0.0  5.1
f1.wav   1         4.0  9.1'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1])
    pd.testing.assert_frame_equal(q1, ans[q1.columns.values])
    d = '''filename sel_id annot_id label  start  end  
f0.wav   0      1             2    3.0  5.1
f0.wav   1      1             2    0.0  2.3
f1.wav   0      0             4    1.0  4.3
f1.wav   0      1             2    4.0  5.1
f1.wav   1      0             4    0.0  0.3
f1.wav   1      1             2    0.0  3.3'''
    ans = pd.read_csv(StringIO(d), delim_whitespace=True, index_col=[0,1,2])
    pd.testing.assert_frame_equal(q2, ans[q2.columns.values])

def test_file_duration_table(five_time_stamped_wave_files):
    """ Test that we can generate a file duration table""" 
    df = st.file_duration_table(five_time_stamped_wave_files)
    d = '''filename,duration
empty_HMS_12_ 5_ 0__DMY_23_ 2_84.wav,0.5
empty_HMS_12_ 5_ 1__DMY_23_ 2_84.wav,0.5
empty_HMS_12_ 5_ 2__DMY_23_ 2_84.wav,0.5
empty_HMS_12_ 5_ 3__DMY_23_ 2_84.wav,0.5
empty_HMS_12_ 5_ 4__DMY_23_ 2_84.wav,0.5'''
    ans = pd.read_csv(StringIO(d))
    pd.testing.assert_frame_equal(df, ans[df.columns.values])
