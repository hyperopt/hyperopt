

if 0:
        for node in dfs(self._vals_memo[expr][0]):
            if node.name == 'literal':
                u_idxs = expr_idxs
                u_vals = scope.asarray(
                        scope.repeat(
                            scope.len(u_idxs),
                            node))
            elif node.name in ('hyperopt_param', 'hyperopt_result'):
                u_idxs = self.idxs_memo[node.arg['obj']]
                u_vals = self.vals_memo[node.arg['obj']]

            elif node in self._idxs_memo:
                idxs_list = self._idxs_memo[node]

                if len(set(idxs_list)) > 1:
                    u_idxs = scope.array_union1(scope.uniq(idxs_list))
                else:
                    u_idxs = idxs_list[0]

                vals_list = self._vals_memo[node]

                if node.name == 'switch':
                    choice = node.pos_args[0]
                    options = node.pos_args[1:]

                    u_choices = scope.idxs_take(
                            self.idxs_memo[choice],
                            self.vals_memo[choice],
                            u_idxs)

                    args_idxs = scope.vchoice_split(u_idxs, u_choices,
                            len(options))
                    u_vals = scope.vchoice_merge(u_idxs, u_choices)
                    for opt_ii, idxs_ii in zip(options, args_idxs):
                        u_vals.pos_args.append(as_apply([
                            self.idxs_memo[opt_ii], self.vals_memo[opt_ii]]))

                    print 'finalizing switch', u_vals

                elif node.name in stoch:
                    # -- this case is separate because we're going to change
                    # the program semantics. If multiple stochastic nodes
                    # are being merged, it means just sample once, and then
                    # index multiple subsets
                    print 'finalizing', node.name
                    assert all(thing.name == node.name for thing in vals_list)

                    # -- assert that all the args except size to each
                    # function are the same
                    vv0 = vals_list[0]
                    vv0d = dict(vv0.arg, rng=None, size=None)
                    for vv in vals_list[1:]:
                        assert vv0d == dict(vv.arg, rng=None, size=None)

                    u_vals = vals_list[0].clone_from_inputs(
                            vals_list[0].inputs())
                    u_vals.set_kwarg('size', scope.len(u_idxs))
                    u_vals.set_kwarg('rng',
                            as_apply(
                                np.random.RandomState(
                                    rng.randint(int(2**30)))))

                else:
                    print 'creating idxs map', node.name
                    u_vals = scope.idxs_map(u_idxs, node.name)
                    u_vals.pos_args.extend(node.pos_args)
                    u_vals.named_args.extend(node.named_args)
                    for arg in node.inputs():
                        u_vals.replace_input(arg,
                                as_apply([
                                    self.idxs_memo[arg],
                                    self.vals_memo[arg]]))

            else:
                print '=' * 80
                print node

            self.idxs_memo[node] = u_idxs
            self.vals_memo[node] = u_vals
