classdef Genetic
    properties
        N {mustBeGreaterThanOrEqual(N, 3)} 
        M 

        pop 
        pop_size {mustBeGreaterThan(pop_size, 3)}
        n_generations = 600 
        prob_mutation = 0.6
        n_parents
        s

        select_parent_method string
        proba_parent_asign_method string
        cross_over_method string
        survivor_select_method string
        mutations string

        best_gens
        min_cost = Inf 

        historic_min = []  
        historic_avg = [] 
    end

    methods
        function obj = Genetic( ...
                N, ...
                pop_size, ...
                select_parent_method, ...
                cross_over_method, ...
                survivor_select_method, ...
                mutations, ...
                proba_parent_asign_method, ...
                n_parents, ...
                s ...
                )
            % Class constructor.
            obj.N = N;
            obj.pop_size = pop_size;

            obj.select_parent_method = select_parent_method;
            obj.cross_over_method = cross_over_method;
            obj.survivor_select_method = survivor_select_method;
            obj.mutations = mutations;

            if nargin >= 7; obj.proba_parent_asign_method = proba_parent_asign_method; end
            if nargin >= 8; obj.n_parents = n_parents; else; obj.n_parents = pop_size/2; end
            if nargin == 9; obj.s = s; else; obj.s = 1.5; end
        end

        function [obj] = initM(obj, min_dist, max_dist, verbose)
            % Function that randomly initializes the cost matrix. This
            % matrix represents the distance between cities, so the main
            % diagonal (dist between a city and itself) should be 0. We
            % also asume that the distance between city i to city j its the
            % same as the distance between city j to city i, so the matrix
            % must be simetric.

            M_lower = tril(randi([min_dist max_dist], obj.N), -1); % Lower triangular matrix
            M_sim = M_lower + M_lower' + eye(obj.N); % Make matrix simetric
            obj.M = M_sim - diag(diag(M_sim)); % Set main diagonal elems to 0

            if verbose
                fprintf('Cost matrix: \n');
                obj.M
            end
        end

        function [obj] = generatePopulation(obj)
            % Function than randomly initializes the initial population.
            % The population is represented as a matrix (pop_size x N),
            % being the pop_size the number of individuals and N the number
            % of cities. The goal is to get the path that cross all
            % cities and return to the initial one traveling the shortest
            % possible path, so the rows must be a permutation without
            % repetition of the number of cities.

            obj.pop = zeros([obj.pop_size obj.N]);
            for i = 1:obj.pop_size
                obj.pop(i, :) = randperm(obj.N); % Permutation encoding
            end
        end

        function [obj, fit] = fitness(obj, population)
            % Function than evaluates the fitness of the population. For
            % this problem we will use a simple function where the cost
            % represents the distance between cities, so the main goal will
            % be to minimize this cost function.
            save = true;
            if nargin > 1
                p = population;
                ps = size(population, 1);
                save = false;
            else
                p = obj.pop;
                ps = obj.pop_size;
            end
            
            cost = zeros([1 ps]);
            for i = 1:ps
                dist = 0;
                for j = 1:obj.N - 1
                    % Distance between first to last city passing through
                    % the other ones
                    dist = dist + obj.M(p(i, j), p(i, j+1));
                end
                % Adding distance between last and first city 
                cost(i) = dist + obj.M(p(i, obj.N), p(i, 1));
            end

            fit = 1./cost; % As less cost greater fitness
            
            if save
                % Adding distance to the historic
                obj.historic_min = [obj.historic_min, min(cost)];
                obj.historic_avg = [obj.historic_avg, mean(cost)];
    
                % Check if the minimun distance until now is in this generation
                if min(cost) < obj.min_cost
                    [min_dist, min_i] = min(cost);
    
                    obj.min_cost = min_dist;
                    obj.best_gens = obj.pop(min_i, :);
                end
            end
        end

        function [parents] = selectParents(obj, fit)
            % Function that select the parents of the next generation
            % according to its adaptation. If the number of parents is not 
            % specifyed, we will use half of the population.

            parents = zeros([obj.n_parents obj.N]);

            switch obj.select_parent_method
                case "roulette"
                    prob = cumsum(fit./sum(fit)); 

                    for i = 1:obj.n_parents
                        p = find(prob < rand, 1);
                        if (isempty(p)); parents(i, :) = obj.pop(1, :); else;  parents(i, :) = obj.pop(p(end)+1, :); end
                    end

                case "rank"
                    if strcmp(obj.proba_parent_asign_method, "lin-rank")
                        p_rank = @(i, s) (2-s)/obj.pop_size + (2*i*(s-1))/(obj.pop_size*(obj.pop_size-1));
                    else % exp-rank
                        p_rank = @(i, s) (1 - exp(-i)) / s; % s instead of c constant to simplify code
                        obj.s = (obj.pop_size*2*(obj.pop_size-1)) / (6*(obj.pop_size-1)+obj.pop_size);
                    end

                    [~, rank] = sort(fit);
                    prob = cumsum(p_rank(rank-1, obj.s));

                    for i = 1:obj.n_parents
                        p = find(prob < rand, 1);
                        if (isempty(p)); parents(i, :) = obj.pop(1, :); else;  parents(i, :) = obj.pop(p(end)+1, :); end
                    end
                    
                otherwise % Tournament 
                    pop_index = 1:1:obj.pop_size;
                    for i = 1:obj.n_parents  
                        if numel(pop_index) < obj.n_parents; n_candidates = numel(pop_index); else; n_candidates = obj.n_parents; end
                        candidates_ind = randperm(numel(pop_index), n_candidates);
                        candidates = pop_index(candidates_ind);
                        [~, rank] = sort(fit(candidates));
                        parents(i, :) = obj.pop(rank(end), :);

                        if strcmp(obj.select_parent_method, "tournament-norep") % Tournament without repetition
                            pop_index(rank(end)) = []; % Delete selected parent from competition
                        end
                    end
            end
        end

        function [childs] = crossover(obj, parents)
            % Function that computes the crossover between the parents.

            childs = zeros([size(parents, 1) obj.N]);

            for i = 1:size(parents, 1) / 2
                p1 = parents((i*2)-1, :);
                p2 = parents(i*2, :);

                switch obj.cross_over_method
                    case 'PMX' % Partially mapped 
                        c1 = pmx_cross(obj, p1, p2);
                        c2 = pmx_cross(obj, p2, p1);

                        childs((i*2)-1, :) = c1;
                        childs(i*2, :) = c2;
    
                    otherwise % cycle
                        c1 = zeros([1 obj.N]);
                        c2 = zeros([1 obj.N]);
                        used_mask = zeros([1 obj.N]);
                        first_pos = 1;
                        cycle = 1;
                        j = 0;

                        while (true)
                            pos = first_pos;
                            while (j~=p1(first_pos))
                                if(mod(cycle, 2) == 1)
                                    c1(pos) = p1(pos);
                                    c2(pos) = p2(pos);
                                elseif (mod(cycle, 2) == 0)
                                    c1(pos) = p2(pos);
                                    c2(pos) = p1(pos);
                                end

                                j = p2(pos);
                                used_mask(pos) = 1;
                                aux = find(p1 == j, 1);
                                pos = aux(1);
                            end

                            cycle = cycle + 1;
                            if (isempty(find(used_mask == 0, 1)))
                                break;
                            else
                                aux = find(used_mask == 0, 1);
                                first_pos = aux(1);
                            end
                        end

                        childs((i*2)-1, :) = c1;
                        childs(i*2, :) = c2;
                end
            end
        end

        function [child] = pmx_cross(obj, p1, p2)
            % Funtion that returns one child after computing the partially
            % mapped crossover method.

            child = zeros([1 obj.N]);

            pivots = sort(randperm(obj.N, 2));
            child(pivots(1):pivots(2)) = p1(pivots(1):pivots(2));
            for i = pivots(1):pivots(2)
                if (p2(i) ~= p1(i) && ~ismember(p2(i), p1(pivots(1):pivots(2))))
                    if (~ismember(p1(i), p2(pivots(1):pivots(2))))
                        child(p2 == p1(i)) = p2(i);
                    else
                         aux = i;
                         while true
                             index = p2 == p1(aux); 
                             if (~ismember(p1(index), p2(pivots(1):pivots(2))))
                                child(p2 == p1(index)) = p2(i);
                                break;
                             end
                             aux = find(index);
                         end
                    end
                end
            end
            for j = 1:obj.N          
                if (child(j) == 0)
                    child(j) = p2(j);
                end
            end
        end

        function [obj] = selectSurvivours(obj, childs)
            % Function that computes the survivours selection method
            switch obj.survivor_select_method
                case 'age'
                    % If for each generation, the children are in the
                    % firsts positions of the population matrix, for each
                    % generation replacement, the elder population will be
                    % at the end of the matrix. This way we only need to
                    % pick the first N (population size) after combine the
                    % childs and then te rest of the original population to
                    % eliminate the elder ones for the next generation.
                    new_pop = [childs; obj.pop];
                    survivours = new_pop(1:obj.pop_size, :); 
                    obj.pop = survivours;
                    
                otherwise % genitor 
                    new_pop = [obj.pop; childs];
                    [~, fit] = fitness(obj, new_pop);
                    [~, rank] = sort(fit,'descend');
                    survivours = new_pop(rank(1:obj.pop_size), :); 
                    obj.pop = survivours;
            end
        end

        function [childs] = mutation(obj, childs)
            % Function that compuetes the mutation by permutation encoding
            % of the genes of the new generation
            
            n_childs = size(childs, 1);

            switch obj.mutations
                case 'exchange'
                    for i = 1:n_childs
                        if (obj.prob_mutation > rand)
                            ind_1 = randi(obj.N); ind_2 = randi(obj.N);
                            aux = childs(i, ind_1);
                            childs(i, ind_1) = childs(i, ind_2);
                            childs(i, ind_2) = aux;
                        end
                    end

                case 'shake'
                    for i = 1:n_childs
                        if (obj.prob_mutation > rand)
                            pivots = sort(randperm(obj.N, 2));   
                            sub_chain = childs(i, pivots(1):pivots(2));
                            perm_index = randperm(numel(sub_chain));
                            shaked = sub_chain(perm_index);
                            childs(i, pivots(1):pivots(2)) = shaked;
                        end
                    end
                otherwise % No mutation
            end
        end

        function [obj] = run(obj, verbose, eartly_stopping_patience)
            tic
            n_no_improve_gens = 0;
            obj = initM(obj, 1, 100, false);
            obj = generatePopulation(obj);
            for gen = 1:obj.n_generations
                [obj, fit] = fitness(obj);
                if nargin > 2 && gen > eartly_stopping_patience
                    if obj.historic_min(end) >= obj.historic_min(end-1); n_no_improve_gens = n_no_improve_gens + 1; else; n_no_improve_gens = 0; end
                    if n_no_improve_gens > eartly_stopping_patience
                        fprintf('Early sttopping at generation: %d\n', gen);
                        obj.n_generations = gen;
                        break;
                    end
                end
                parents = selectParents(obj, fit);
                childs = crossover(obj, parents);
                childs = mutation(obj, childs);
                obj = selectSurvivours(obj, childs);
            end
            time = toc;

            if verbose > 0
                fprintf('Genetic configuration: \n');
                fprintf('\t-%s\n', obj.select_parent_method);
                if strcmp(obj.select_parent_method, "rank")
                    fprintf('\t\t-%s\n', obj.proba_parent_asign_method); 
                    if strcmp(obj.proba_parent_asign_method, "lin-rank"); fprintf('\t\t-s=%.1f\n', obj.s); end
                end
                fprintf('\t-%s\n', obj.cross_over_method);
                fprintf('\t-%s\n', obj.survivor_select_method);
                fprintf('\t-%s\n', obj.mutations);
                fprintf('\t-mutation_prob: %.1f\n', obj.prob_mutation);
                fprintf('\t-n_parents %d\n', obj.n_parents);
                fprintf('Took %f[s] to execute.\n', time);
                [~, best_generation] = min(obj.historic_min);
                fprintf('Best gens reached at generation: %d\n', best_generation);
                fprintf('Best cost %f\n', obj.min_cost);
                fprintf('Path: ')
                for i = 1:obj.N
                    fprintf('%d ', obj.best_gens(i));
                end
                fprintf('\n\n\n');
                if verbose == 2; printCurve(obj, false); elseif verbose == 3; printCurve(obj, true); end
            end
        end

        function printCurve(obj, save)
            figure;
            plot(1:obj.n_generations, obj.historic_min);
            hold on;
            plot(1:obj.n_generations, obj.historic_avg);
            title(['spm:' obj.select_parent_method ...
                ' com:' obj.cross_over_method ...
                ' ssm:' obj.survivor_select_method ...
                ' m:' obj.mutations ...
                ' n_parents' obj.n_parents ...
            ])
            if strcmp(obj.select_parent_method, "rank")
                if strcmp(obj.proba_parent_asign_method, "lin-rank")
                    title(['spm:' obj.select_parent_method ...
                        ' ppam:' obj.proba_parent_asign_method, ...
                        ' s:' obj.s, ...
                        ' com:' obj.cross_over_method ...
                        ' ssm:' obj.survivor_select_method ...
                        ' m:' obj.mutations ...
                        ' n_parents' obj.n_parents ...
                        ])
                else
                    title(['spm:' obj.select_parent_method ...
                        ' ppam:' obj.proba_parent_asign_method, ...
                        ' s:' obj.s, ...
                        ' com:' obj.cross_over_method ...
                        ' ssm:' obj.survivor_select_method ...
                        ' m:' obj.mutations ...
                        ' n_parents' obj.n_parents ...
                        ])
                end
            end
            xlabel('Genetarion');
            ylabel('Distance');
            legend('min', 'avg');
            filename = 'Figures\final.fig';
            if save; savefig(filename); end
        end
    end
end