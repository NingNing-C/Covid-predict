from typing import Any, Set


class GeneticAttacker(ClassificationAttacker):

    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self, 
            pop_size : int = 20, 
            max_iters : int = 20, 
            tokenizer : Optional[Tokenizer] = None, 
            substitute : Optional[WordSubstitute] = None, 
            lang = None,
            filter_words : List[str] = None
        ):
        """
        Generating Natural Language Adversarial Examples. Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang. EMNLP 2018.
        `[pdf] <https://www.aclweb.org/anthology/D18-1316.pdf>`__
        `[code] <https://github.com/nesl/nlp_adversarial_examples>`__
        
        Args:
            pop_size: Genetic algorithm popluation size. **Default:** 20
            max_iter: Maximum generations of genetic algorithm. **Default:** 20
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            filter_words: A list of words that will be preserved in the attack procesudre.

        :Classifier Capacity:
            * get_pred
            * get_prob
        
        """
        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if substitute is not None:
            lst.append(substitute)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)
        
        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute
        self.pop_size = pop_size
        self.max_iters = max_iters

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        check_language([self.tokenizer, self.substitute], self.__lang_tag)


    def attack(self, victim: Classifier, x_orig, goal: ClassifierGoal):
        #x_orig = x_orig.lower()
        
        x_orig = self.tokenizer.tokenize(x_orig)
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        neighbours_nums = [
            self.get_neighbour_num(word, pos) if word not in self.filter_words else 0
            for word, pos in zip(x_orig, x_pos)
        ]
        neighbours = [
            self.get_neighbours(word, pos)
            if word not in self.filter_words
            else []
            for word, pos in zip(x_orig, x_pos)
        ]

        if np.sum(neighbours_nums) == 0:
            return None
        w_select_probs = neighbours_nums / np.sum(neighbours_nums)

        pop = [  # generate population
            self.perturb(
                victim, x_orig, x_orig, neighbours, w_select_probs, goal
            )
            for _ in range(self.pop_size)
        ]
        for i in range(self.max_iters):
            pop_preds = victim.get_prob(self.make_batch(pop))
            ## record the populations 
            outpath_seq='/home/chenn0a/chenn0a/covid_esm1b/attack_result/covid/genetic_omicron_exp/pop_seq'
            outpath_seq = f'{outpath_seq}-{os.getpid()}'
            outpath_pre='/home/chenn0a/chenn0a/covid_esm1b/attack_result/covid/genetic_omicron_exp/pop_pre'
            outpath_pre = f'{outpath_pre}-{os.getpid()}'
            with open(outpath_seq,'a+') as fp:
                for item in pop:
                    fp.write("%s\n" % ''.join(item))
            with open(outpath_pre,'a+') as f2:
                for item in pop_preds[:,1]:
                    f2.write("%s\n" % item)
            if goal.targeted:
                top_attack = np.argmax(pop_preds[:, goal.target])
                if np.argmax(pop_preds[top_attack, :]) == goal.target:
                    return self.tokenizer.detokenize(pop[top_attack])
            else:
                top_attack = np.argmax(-pop_preds[:, goal.target])
                if np.argmax(pop_preds[top_attack, :]) != goal.target and i>=1:
                    return self.tokenizer.detokenize(pop[top_attack])

            pop_scores = pop_preds[:, goal.target]
            if not goal.targeted:
                pop_scores = 1.0 - pop_scores

            if np.sum(pop_scores) == 0:
                return None
            pop_scores = pop_scores / np.sum(pop_scores)

            elite = [pop[top_attack]]
            ## remove pop score of p1
            parent_indx_1 = np.random.choice(
                self.pop_size, size=self.pop_size - 1
            )
            parent_indx_2 = np.random.choice(
                self.pop_size, size=self.pop_size - 1
            )
            childs = [
                self.crossover(pop[p1], pop[p2])
                for p1, p2 in zip(parent_indx_1, parent_indx_2)
            ]
            childs = [
                self.perturb(
                    victim, x_cur, x_orig, neighbours, w_select_probs, goal
                )
                for x_cur in childs
            ]
            pop = elite + childs

        return None  # Failed

    def get_neighbour_num(self, word, pos):
        try:
            return len(self.substitute(word, pos))
        except WordNotInDictionaryException:
            return 0

    def get_neighbours(self, word, pos):
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.substitute(word, pos),
                )
            )
        except WordNotInDictionaryException:
            return []

    def select_best_replacements(
        self, clsf, indx, neighbours, x_cur, x_orig, goal : ClassifierGoal
    ):
        def do_replace(word):
            ret = x_cur.copy()
            ret[indx] = word
            return ret
        new_list = []
        rep_words = []
        for word in neighbours:
            if word != x_orig[indx]:
                new_list.append(do_replace(word))
                rep_words.append(word)
        if len(new_list) == 0:
            return x_cur
        new_list.append(x_cur)

        pred_scores = clsf.get_prob(self.make_batch(new_list))[:, goal.target]
        if goal.targeted:
            new_scores = pred_scores[:-1] - pred_scores[-1]
        else:
            new_scores = pred_scores[-1] - pred_scores[:-1]

        if np.max(new_scores) > 0:
            return new_list[np.argmax(new_scores)]
        else:
            return x_cur

    def select_top_replacements(
        self, clsf, indx, neighbours, x_cur, x_orig, goal : ClassifierGoal
    ):
        def do_replace(word):
            ret = x_cur.copy()
            ret[indx] = word
            return ret
        new_list = []
        rep_words = []
        for word in neighbours:
            if word != x_orig[indx]:
                new_list.append(do_replace(word))
                rep_words.append(word)
        if len(new_list) == 0:
            return x_cur
        new_list.append(x_cur)

        pred_scores = clsf.get_prob(self.make_batch(new_list))[:, goal.target]
        if goal.targeted:
            new_scores = pred_scores[:-1] - pred_scores[-1]
        else:
            new_scores = pred_scores[-1] - pred_scores[:-1]

        
        top_seq=sorted(zip(new_scores,new_list[:-1]),reverse=True)[:5]
        top_seq.append((0,x_cur))
        return random.choice(top_seq)[1]
        

    def make_batch(self, sents):
        return [self.tokenizer.detokenize(sent) for sent in sents]

    def perturb(
        self, clsf, x_cur, x_orig, neighbours, w_select_probs, goal : ClassifierGoal
    ):
        x_len = len(x_cur)
        num_mods = 0
        for i in range(x_len):
            if x_cur[i] != x_orig[i]:
                num_mods += 1
        np.random.seed(8089)
        mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        ## for bebug
        #mod_idx=26
        if num_mods < np.sum(
            np.sign(w_select_probs)
        ):  # exists at least one indx not modified
            while x_cur[mod_idx] != x_orig[mod_idx]:  # already modified
                mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[
                    0
                ]  # random another indx
                
                #mod_idx=26
        ## remove the impact of seed
        import time
        t = 1000 * time.time() # current time in milliseconds
        np.random.seed(int(t) % 2**32)

        return self.select_top_replacements(
            clsf, mod_idx, neighbours[mod_idx], x_cur, x_orig, goal
        )

    def crossover(self, x1, x2):
        ret = []
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                ret.append(x1[i])
            else:
                ret.append(x2[i])
        return ret
class ClassificationAttacker(Attacker):
    """
    The base class of all classification attackers.
    """

    def __call__(self, victim: Classifier, input_: Any):
        if not isinstance(victim, Classifier):
            raise TypeError("`victim` is an instance of `%s`, but `%s` expected" % (victim.__class__.__name__, "Classifier"))
        if Tag("get_pred", "victim") not in victim.TAGS:
            raise AttributeError("`%s` needs victim to support `%s` method" % (self.__class__.__name__, "get_pred"))
        self._victim_check(victim)


        if "target" in input_:
            goal = ClassifierGoal(input_["target"], targeted=True)
        else:
            origin_x = victim.get_pred([ input_["x"] ])[0]
            goal = ClassifierGoal( origin_x, targeted=False )
        
        adversarial_sample = self.attack(victim, input_["x"], goal)

        if adversarial_sample is not None:
            y_adv = victim.get_pred([ adversarial_sample ])[0]
            if not goal.check( adversarial_sample, y_adv ):
                raise RuntimeError("Check attacker result failed: result ([%d] %s) expect (%s%d)" % ( y_adv, adversarial_sample, "" if goal.targeted else "not ", goal.target))
        return adversarial_sample

class AttackGoal:
    def check(self, adversarial_sample, prediction) -> bool:
        raise NotImplementedError()
class ClassifierGoal(AttackGoal):
    def __init__(self, target, targeted):
        self.target = target
        self.targeted = targeted
    
    @property
    def is_targeted(self):
        return self.targeted

    def check(self, adversarial_sample, prediction):
        if self.targeted:
            return prediction == self.target
        else:
            return prediction != self.target
class Tag(object):
    def __init__(self, tag_name : str, type_ = None):
        self.__tag_name = tag_name
        self.__type : str = type_ if type_ is not None else ""
    
    @property
    def type(self) -> str:
        return self.__type
    
    @property
    def name(self) -> str:
        return self.__tag_name
    
    def __str__(self) -> str:
        return self.type + ":" + self.__tag_name
    
    def __eq__(self, o: object):
        return str(o).lower() == str(self).lower()
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __repr__(self) -> str:
        return "<%s>" % str(self)

class Attacker:
    """
    The base class of all attackers.
    """


    TAGS : Set[Tag] = set()

    def __call__(self, victim : Victim, input_ : Any):
        raise NotImplementedError()
    
    def _victim_check(self, victim : Victim):
        lang = victim.supported_language
        if lang is not None and lang not in self.TAGS:
            available_langs = []
            for it in self.TAGS:
                if it.type == "lang":
                    available_langs.append(it.name)
            raise RuntimeError("Victim supports language `%s` but `%s` expected." % (lang.name, available_langs))
        
        for tag in self.TAGS:
            if tag.type == "victim":
                if tag not in victim.TAGS:
                    raise AttributeError("`%s` needs victim to support `%s` method" % (self.__class__.__name__, tag.name))

    def attack(self, victim : Victim, input_ : Any, goal : AttackGoal):
        raise NotImplementedError()


class AttackEval:
    def __init__(self,
        attacker : Attacker,
        victim : Victim,
        language : Optional[str] = None,
        tokenizer : Optional[Tokenizer] = None,
        invoke_limit : Optional[int] = None,
        metrics : List[Union[AttackMetric, MetricSelector]] = []
    ):
        """
        `AttackEval` is a class used to evaluate attack metrics in OpenAttack.

        Args:
            attacker: An attacker, must be an instance of :py:class:`.Attacker` .
            victim: A victim model, must be an instance of :py:class:`.Vicitm` .
            language: The language used for the evaluation. If is `None` then `AttackEval` will intelligently select the language based on other parameters.
            tokenizer: A tokenizer used for visualization.
            invoke_limit: Limit on the number of model invokes.
            metrics: A list of metrics. Each element must be an instance of :py:class:`.AttackMetric` or :py:class:`.MetricSelector` .

        """

        if language is None:
            lst = [attacker]
            if tokenizer is not None:
                lst.append(tokenizer)
            if victim.supported_language is not None:
                lst.append(victim)
            for it in metrics:
                if isinstance(it, AttackMetric):
                    lst.append(it)

            lang_tag = get_language(lst)
        else:
            lang_tag = language_by_name(language)
            if lang_tag is None:
                raise ValueError("Unsupported language `%s` in attack eval" % language)

        self._tags = { lang_tag }

        if tokenizer is None:
            self.tokenizer = get_default_tokenizer(lang_tag)
        else:
            self.tokenizer = tokenizer

        self.attacker = attacker
        self.victim = victim
        self.metrics = []
        for it in metrics:
            if isinstance(it, MetricSelector):
                v = it.select(lang_tag)
                if v is None:
                    raise RuntimeError("`%s` does not support language %s" % (it.__class__.__name__, lang_tag.name))
                self.metrics.append( v )
            elif isinstance(it, AttackMetric):
                self.metrics.append( it )
            else:
                raise TypeError("`metrics` got %s, expect `MetricSelector` or `AttackMetric`" % it.__class__.__name__)
        self.invoke_limit = invoke_limit
        
    @property
    def TAGS(self):
        return self._tags
    
    def __measure(self, data, adversarial_sample):
        ret = {}
        for it in self.metrics:
            value = it.after_attack(data, adversarial_sample)
            if value is not None:
                ret[it.name] = value
        return ret


    def __iter_dataset(self, dataset):
        for data in dataset:
            v = data
            for it in self.metrics:
                ret = it.before_attack(v)
                if ret is not None:
                    v = ret
            yield v
    
    def __iter_metrics(self, iterable_result):
        for data, result in iterable_result:
            adversarial_sample, attack_time, invoke_times = result
            ret = {
                "data": data,
                "success": adversarial_sample is not None,
                "result": adversarial_sample,
                "metrics": {
                    "Running Time": attack_time,
                    "Query Exceeded": self.invoke_limit is not None and invoke_times > self.invoke_limit,
                    "Victim Model Queries": invoke_times,
                    ** self.__measure(data, adversarial_sample)
                }
            }
            yield ret

    def ieval(self, dataset : Iterable[Dict[str, Any]], num_workers : int = 0, chunk_size : Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Iterable evaluation function of `AttackEval` returns an Iterator of result.

        Args:
            dataset: An iterable dataset.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Yields:
            A dict contains the result of each input samples.

        """

        if num_workers > 0:
            ctx = mp.get_context("spawn")
            if chunk_size is None:
                chunk_size = num_workers
            with ctx.Pool(num_workers, initializer=worker_init, initargs=(self.attacker, self.victim, self.invoke_limit)) as pool:
                for ret in self.__iter_metrics(zip(dataset, pool.imap(worker_process, self.__iter_dataset(dataset), chunksize=chunk_size))):
                    yield ret
                   
        else:
            def result_iter():
                for data in self.__iter_dataset(dataset):
                    yield attack_process(self.attacker, self.victim, data, self.invoke_limit)
            for ret in self.__iter_metrics(zip(dataset, result_iter())):
                yield ret

    def eval(self, dataset: Iterable[Dict[str, Any]], s_file:str='s_seq.txt',f_file:str='f_seq.txt', total_len : Optional[int] = None, visualize : bool = False, progress_bar : bool = False, num_workers : int = 0, chunk_size : Optional[int] = None):
        """
        Evaluation function of `AttackEval`.

        Args:
            dataset: An iterable dataset.
            total_len: Total length of dataset (will be used if dataset doesn't has a `__len__` attribute).
            visualize: Display a pretty result for each data in the dataset.
            progress_bar: Display a progress bar if `True`.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Returns:
            A dict of attack evaluation summaries.

        """


        if hasattr(dataset, "__len__"):
            total_len = len(dataset)
        
        def tqdm_writer(x):
            return tqdm.write(x, end="")
        
        if progress_bar:
            result_iterator = tqdm(self.ieval(dataset, num_workers, chunk_size), total=total_len)
        else:
            result_iterator = self.ieval(dataset, num_workers, chunk_size)

        total_result = {}
        total_result_cnt = {}
        total_inst = 0
        success_inst = 0

        # Begin for
        for i, res in enumerate(result_iterator):
            total_inst += 1
            success_inst += int(res["success"])

            if visualize and (TAG_Classification in self.victim.TAGS):
                x_orig = res["data"]["x"]
                if res["success"]:
                    x_adv = res["result"]
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([x_orig, x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                        y_adv = probs[1]
                        with open(s_file,'a+') as fs:
                            fs.write(x_orig+'\t'+str(y_orig[0])+'\t'+str(y_orig[1])+'\t'+x_adv+'\t'+str(y_adv[0])+'\t'+str(y_adv[1])+'\n')
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig, x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                        y_adv = int(preds[1])
                        with open(s_file,'a+') as fs:
                            fs.write(x_orig+'\t'+str(y_orig[0])+'\t'+str(y_orig[1])+'\t'+x_adv+'\t'+str(y_adv[0])+'\t'+str(y_adv[1])+'\n')
                    else:
                        raise RuntimeError("Invalid victim model")
                else:
                    y_adv = None
                    x_adv = None
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                        with open(f_file,'a+') as ff:
                            ff.write(x_orig+'\t'+str(y_orig[0])+'\t'+str(y_orig[1])+'\n')
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                        with open(f_file,'a+') as ff:
                            ff.write(x_orig+'\t'+str(y_orig[0])+'\t'+str(y_orig[1])+'\n')
                    else:
                        raise RuntimeError("Invalid victim model")
                info = res["metrics"]
                info["Succeed"] = res["success"]
                if progress_bar:
                    visualizer(i + 1, x_orig, y_orig, x_adv, y_adv, info, tqdm_writer, self.tokenizer)
                else:
                    visualizer(i + 1, x_orig, y_orig, x_adv, y_adv, info, sys.stdout.write, self.tokenizer)
            for kw, val in res["metrics"].items():
                if val is None:
                    continue

                if kw not in total_result_cnt:
                    total_result_cnt[kw] = 0
                    total_result[kw] = 0
                total_result_cnt[kw] += 1
                total_result[kw] += float(val)
        # End for

        summary = {}
        summary["Total Attacked Instances"] = total_inst
        summary["Successful Instances"] = success_inst
        summary["Attack Success Rate"] = success_inst / total_inst
        for kw in total_result_cnt.keys():
            if kw in ["Succeed"]:
                continue
            if kw in ["Query Exceeded"]:
                summary["Total " + kw] = total_result[kw]
            else:
                summary["Avg. " + kw] = total_result[kw] / total_result_cnt[kw]
        
        if visualize:
            result_visualizer(summary, sys.stdout.write)
        return summary