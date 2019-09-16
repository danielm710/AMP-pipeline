import luigi
import os
from subprocess import Popen, PIPE
import json
import pickle
import pandas as pd
from sklearn.externals import joblib

# Import custom scripts
from scripts import read_fasta
from scripts import get_reduced_form
from scripts import reduced_matrix
from scripts import generate_profile
from scripts import feature_processing
from scripts import model

out_dir = "output"
luigi.configuration.add_config_path('/home/danielm710/project/luigi/train_amp.cfg')

def run_cmd(cmd):
    p = Popen(cmd, stdout=PIPE)
    output = p.communicate()[0]
    return output

class reduced_form(luigi.Config):
    is_reduced = luigi.Parameter()
    dayhoff = luigi.Parameter()
    diamond = luigi.Parameter()
    murphy_4 = luigi.Parameter()
    murphy_8 = luigi.Parameter()
    murphy_10 = luigi.Parameter()

class Read_fasta(luigi.Task):
    pos_fasta = luigi.Parameter()
    neg_fasta = luigi.Parameter()

    train_preprocessed_dir = os.path.join(out_dir, "train_preprocessed")

    def output(self):
        output = {}
        tmp = {}

        pos_dict = os.path.join(self.train_preprocessed_dir, "pos_dict.json")
        neg_dict = os.path.join(self.train_preprocessed_dir, "neg_dict.json")

        output["pos"] = luigi.LocalTarget(pos_dict)
        output["neg"] = luigi.LocalTarget(neg_dict)

        return output

    def run(self):
        # Make output directory if not exist
        run_cmd(['mkdir',
                '-p',
                self.train_preprocessed_dir])

        # Read fasta
        pos_dict = read_fasta.read_fasta(self.pos_fasta)
        neg_dict = read_fasta.read_fasta(self.neg_fasta)

        with self.output()["pos"].open('w') as fh:
            json.dump(pos_dict, fh)

        with self.output()["neg"].open('w') as fh:
            json.dump(neg_dict, fh)

class Convert_to_reduced(luigi.Task):
    train_preprocessed_dir = os.path.join(out_dir, "train_preprocessed")

    def requires(self):
        return Read_fasta()

    def output(self):
        output = {}

        pos_reduced = os.path.join(self.train_preprocessed_dir,
                "pos_reduced.pkl")
        neg_reduced = os.path.join(self.train_preprocessed_dir,
                "neg_reduced.pkl")

        # Adding "luigi.format.Nop" opens the file in binary mode
        output["pos"] = luigi.LocalTarget(pos_reduced, format=luigi.format.Nop)
        output["neg"] = luigi.LocalTarget(neg_reduced, format=luigi.format.Nop)

        return output

    def run(self):
        # Make output directory if not exist
        run_cmd(['mkdir',
                '-p',
                self.train_preprocessed_dir])

        # Load input protein dictionary
        with self.input()['pos'].open('r') as fh:
            pos_dict = json.load(fh)

        with self.input()['neg'].open('r') as fh:
            neg_dict = json.load(fh)

        # Make pandas dataframe
        pos_df = pd.DataFrame(pos_dict.items(),
                                columns=['seq_ID' ,'sequence'])\
                                        .set_index('seq_ID')\
                                        #.reset_index()

        neg_df = pd.DataFrame(neg_dict.items(),
                                columns=['seq_ID' ,'sequence'])\
                                        .set_index('seq_ID')\
                                        #.reset_index()

        # Convert to reduced format
        method_dict = {"dayhoff": reduced_form().dayhoff,
                    "diamond": reduced_form().diamond,
                    "murphy_4": reduced_form().murphy_4,
                    "murphy_8": reduced_form().murphy_8,
                    "murphy_10": reduced_form().murphy_10}

        # Get reduced matrix
        reduced_matrices = {}

        for method in method_dict:
            reduced_matrices[method] = reduced_matrix.\
                                        get_reduced_matrix(method)

        pos_reduced = get_reduced_form.convert_to_reduced(
                pos_df,
                reduced_form().is_reduced,
                method_dict,
                reduced_matrices)

        neg_reduced = get_reduced_form.convert_to_reduced(
                neg_df,
                reduced_form().is_reduced,
                method_dict,
                reduced_matrices)

        # Save output
        with self.output()['pos'].open('w') as fh:
            pickle.dump(pos_reduced, fh)
        with self.output()['neg'].open('w') as fh:
            pickle.dump(neg_reduced, fh)

class Get_kmers_table(luigi.Task):
    core = luigi.IntParameter(default=1)

    def requires(self):
        return Convert_to_reduced()

    def output(self):
        misc_dir = os.path.join(out_dir, "miscellaneous")

        kmers_table = os.path.join(misc_dir, "kmers_table.pkl")
        return luigi.LocalTarget(kmers_table, format=luigi.format.Nop)

    def run(self):
        method_dict = {"dayhoff": reduced_form().dayhoff,
                    "diamond": reduced_form().diamond,
                    "murphy_4": reduced_form().murphy_4,
                    "murphy_8": reduced_form().murphy_8,
                    "murphy_10": reduced_form().murphy_10}

        reduced_matrices = {}

        for method in method_dict:
            reduced_matrices[method] = reduced_matrix.\
                                        get_reduced_matrix(method)

        kmers_dict = generate_profile.compute_kmer_table(
                reduced_form().is_reduced,
                method_dict,
                reduced_matrices,
                self.core)

        with self.output().open('w') as fh:
            pickle.dump(kmers_dict, fh)

class Calculate_relative_frequency_profile(luigi.Task):
    train_preprocessed_dir = os.path.join(out_dir, "train_preprocessed")

    core = luigi.IntParameter(default=1)

    def requires(self):
        return {
                "reduced_form": Convert_to_reduced(),
                "kmers_table": Get_kmers_table()
                }

    def output(self):
        pos_relative_profile = os.path.join(self.train_preprocessed_dir,
                "pos_rel_freq_profile.csv")
        neg_relative_profile = os.path.join(self.train_preprocessed_dir,
                "neg_rel_freq_profile.csv")

        output = {}
        output['pos'] = luigi.LocalTarget(pos_relative_profile)
        output['neg'] = luigi.LocalTarget(neg_relative_profile)

        return output

    def run(self):
        # Load inputs
        with self.input()["kmers_table"].open('r') as fh:
            kmers_tables = pickle.load(fh)
        with self.input()["reduced_form"]["pos"].open('r') as fh:
            pos_reduced = pickle.load(fh)
        with self.input()["reduced_form"]["neg"].open('r') as fh:
            neg_reduced = pickle.load(fh)

        pos_reduced_with_len = generate_profile.calculate_length(pos_reduced)
        neg_reduced_with_len = generate_profile.calculate_length(neg_reduced)

        pos_rel_freq_df = generate_profile.calculate_relative_frequency(
                pos_reduced_with_len,
                kmers_tables,
                self.core
                )

        neg_rel_freq_df = generate_profile.calculate_relative_frequency(
                neg_reduced_with_len,
                kmers_tables,
                self.core
                )

        with self.output()['pos'].open('w') as fh:
            pos_rel_freq_df.to_csv(fh)

        with self.output()['neg'].open('w') as fh:
            neg_rel_freq_df.to_csv(fh)

class Mann_Whitney_U_test(luigi.Task):
    multiple_comparison_method = luigi.Parameter(default="bonferroni")
    adjusted_pval = luigi.FloatParameter(default=0.001)
    threshold = luigi.FloatParameter(default=0.3)

    def requires(self):
        return Calculate_relative_frequency_profile()

    def output(self):
        misc_dir = os.path.join(out_dir, "miscellaneous")
        Mann_Whitney_kmers = os.path.join(misc_dir, "mann_whitney_kmers.txt")

        return luigi.LocalTarget(Mann_Whitney_kmers)

    def run(self):
        # Load input
        with self.input()['pos'].open('r') as fh:
            pos_rel_freq_df = pd.read_csv(fh).set_index('seq_ID')
        with self.input()['neg'].open('r') as fh:
            neg_rel_freq_df = pd.read_csv(fh).set_index('seq_ID')

        mw_kmers = feature_processing.Mann_Whitney_U_test(
                pos_rel_freq_df,
                neg_rel_freq_df,
                self.multiple_comparison_method,
                self.adjusted_pval,
                self.threshold
                )

        with self.output().open('w') as fh:
            for kmer in mw_kmers:
                fh.write(kmer + "\n")

class Welch_t_test(luigi.Task):
    multiple_comparison_method = luigi.Parameter(default="bonferroni")
    adjusted_pval = luigi.FloatParameter(default=0.001)
    threshold = luigi.FloatParameter(default=0.3)

    def requires(self):
        return Calculate_relative_frequency_profile()

    def output(self):
        misc_dir = os.path.join(out_dir, "miscellaneous")
        Welch_kmers = os.path.join(misc_dir, "welch_kmers.txt")

        return luigi.LocalTarget(Welch_kmers)

    def run(self):
        # Load input
        with self.input()['pos'].open('r') as fh:
            pos_rel_freq_df = pd.read_csv(fh).set_index('seq_ID')
        with self.input()['neg'].open('r') as fh:
            neg_rel_freq_df = pd.read_csv(fh).set_index('seq_ID')

        welch_kmers = feature_processing.Welch_t_test(
                pos_rel_freq_df,
                neg_rel_freq_df,
                self.multiple_comparison_method,
                self.adjusted_pval,
                self.threshold
                )

        with self.output().open('w') as fh:
            for kmer in welch_kmers:
                fh.write(kmer + "\n")

class Get_intersection_kmers(luigi.Task):

    def requires(self):
        return {
                "Mann_Whitney": Mann_Whitney_U_test(),
                "Welch": Welch_t_test()
                }

    def output(self):
        misc_dir = os.path.join(out_dir, "miscellaneous")
        intersection_kmers = os.path.join(misc_dir, "intersection_kmers.txt")

        return luigi.LocalTarget(intersection_kmers)

    def run(self):
        # Load input
        mw_kmers = set()
        welch_kmers = set()
        with self.input()["Mann_Whitney"].open('r') as fh:
            for line in fh:
                mw_kmers.add(line.strip())

        with self.input()["Welch"].open('r') as fh:
            for line in fh:
                welch_kmers.add(line.strip())

        intersection_kmers = mw_kmers.intersection(welch_kmers)

        with self.output().open('w') as fh:
            for kmer in intersection_kmers:
                fh.write(kmer + "\n")

class Remove_correlated_kmers(luigi.Task):
    threshold = luigi.FloatParameter(default=0.9)

    def requires(self):
        return {
                "relative_freq": Calculate_relative_frequency_profile(),
                "kmers": Get_intersection_kmers()
                }

    def output(self):
        misc_dir = os.path.join(out_dir, "miscellaneous")
        candidate_kmers = os.path.join(misc_dir, "final_candidate_kmers.txt")

        return luigi.LocalTarget(candidate_kmers)

    def run(self):
        # Load input
        intersection_kmers = []
        with self.input()["kmers"].open('r') as fh:
            for line in fh:
                intersection_kmers.append(line.strip())

        with self.input()["relative_freq"]["pos"].open('r') as fh:
            pos_freq_df = pd.read_csv(fh).set_index('seq_ID')[intersection_kmers]

        with self.input()["relative_freq"]["neg"].open('r') as fh:
            neg_freq_df = pd.read_csv(fh).set_index('seq_ID')[intersection_kmers]

        candidate_kmers = feature_processing.remove_correlated_features(
                pos_freq_df,
                neg_freq_df,
                self.threshold
                )

        with self.output().open('w') as fh:
            for kmer in candidate_kmers:
                fh.write(kmer + "\n")

class Get_training_data(luigi.Task):

    def requires(self):
        return {
                "kmers": Remove_correlated_kmers(),
                "relative_freq": Calculate_relative_frequency_profile()
                }

    def output(self):
        model_dir = os.path.join(out_dir, "model")
        train_data = os.path.join(model_dir, "train_data.csv")
        train_label = os.path.join(model_dir, "train_label.csv")

        output = {
                "train_data": luigi.LocalTarget(train_data),
                "train_label": luigi.LocalTarget(train_label)
                }

        return output

    def run(self):
        # Load input
        candidate_kmers = []
        with self.input()["kmers"].open('r') as fh:
            for line in fh:
                candidate_kmers.append(line.strip())

        # Extract candidate kmers from relative frequency profile
        with self.input()["relative_freq"]["pos"].open('r') as fh:
            pos_rel_freq_df = pd.read_csv(fh).set_index('seq_ID')[candidate_kmers]

        with self.input()["relative_freq"]["neg"].open('r') as fh:
            neg_rel_freq_df = pd.read_csv(fh).set_index('seq_ID')[candidate_kmers]

        # Sort columns to have consistent column order
        pos_rel_freq_df = pos_rel_freq_df.reindex(
                sorted(pos_rel_freq_df.columns), axis=1
                )
        neg_rel_freq_df = neg_rel_freq_df.reindex(
                sorted(neg_rel_freq_df.columns), axis=1
                )

        # Add labels
        pos_rel_freq_df["label"] = 1
        neg_rel_freq_df["label"] = 0

        # Shuffle samples
        pos_rel_freq_df = pos_rel_freq_df.sample(frac=1)
        neg_rel_freq_df = neg_rel_freq_df.sample(frac=1)

        # Visualize data
        print("----------Visualizing Positive Data----------")
        print(pos_rel_freq_df.head(3))
        print("--------------------------------------------")
        print("----------Visualizing Negative Data----------")
        print(neg_rel_freq_df.head(3))
        print("--------------------------------------------")

        # Combine positive and negative df
        train = pd.concat([pos_rel_freq_df, neg_rel_freq_df])
        train_data = train.iloc[:, :-1]
        train_label = train.iloc[:, -1:]

        # Write to a file
        with self.output()["train_data"].open('w') as fh:
            train_data.to_csv(fh)
        with self.output()["train_label"].open('w') as fh:
            train_label.to_csv(fh)

class Fit_model(luigi.Task):
    core = luigi.IntParameter(default=1)
    score = luigi.Parameter(default="recall")
    k_cv = luigi.IntParameter(default=5)

    def requires(self):
        return Get_training_data()

    def output(self):
        model_dir = os.path.join(out_dir, "model")
        model = os.path.join(model_dir, "randomforest.sav")

        return luigi.LocalTarget(model, format=luigi.format.Nop)

    def run(self):
        with self.input()["train_data"].open('r') as fh:
            train_data = pd.read_csv(fh).set_index('seq_ID')
        with self.input()["train_label"].open('r') as fh:
            train_label = pd.read_csv(fh).set_index('seq_ID')

        rf_fit = model.fit_model(
                train_data,
                train_label,
                self.score,
                self.k_cv,
                self.core
                )

        with self.output().open('w') as fh:
            joblib.dump(rf_fit, fh)

class Feature_importance(luigi.Task):

    def requires(self):
        return {
                "train": Get_training_data(),
                "model": Fit_model()
                }

    def output(self):
        model_dir = os.path.join(out_dir, "model")

        feature_importance = os.path.join(model_dir, "feature_importance.csv")

        return luigi.LocalTarget(feature_importance)

    def run(self):
        # Load input
        with self.input()["train"]["train_data"].open('r') as fh:
            train_data = pd.read_csv(fh).set_index('seq_ID')

        with self.input()["model"].open('r') as fh:
            rf_fit = joblib.load(fh)

        feature_importance = model.get_feature_importance(
                train_data,
                rf_fit
                )

        with self.output().open('w') as fh:
            feature_importance.to_csv(fh)


#dummy class to run all the tasks
class run_tasks(luigi.Task):
    def requires(self):
        task_list = [Read_fasta(),
                    Convert_to_reduced(),
                    Get_kmers_table(),
                    Calculate_relative_frequency_profile(),
                    Mann_Whitney_U_test(),
                    Welch_t_test(),
                    Get_intersection_kmers(),
                    Remove_correlated_kmers(),
                    Get_training_data(),
                    Fit_model(),
                    Feature_importance()]

        return task_list

if __name__ == '__main__':
    luigi.run()

