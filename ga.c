/**
 * 遺伝的アルゴリズムを用いた例：巡回セールスマン問題を解く
 * Written by M.Nagaku 1999
 * 無断使用、改変を奨励し(^^;、使用に制限は設けないが
 * 保証は一切行わないものとする。
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <memory.h>

#define GA_PROGRAM
#define RAND
#define TOWN_COUNT			192						// 都市の数。この例では15都市。
#define POPULATION_COUNT	5000					// 集団の大きさ、個体数。
#define CUT_LINE			500						// 足切りライン。この順位以下の個体は淘汰される。
#define MUTATION_ODDS		0.05					// 突然変異を起こす確率。
#define STOP_COST			90						// このコスト以下の解を得た時点で終了する。
#define MAX_COST			20						// 都市間の設定コストをこの値未満とする。
#define PENALTY				TOWN_COUNT * MAX_COST	// ペナルティ。都市間が無接続時のコスト、非通過都市に関するペナルティ。

/* map：都市間の移動にかかるコストの行列表現。
 *      この例ではやってないが、この表現なら往路、復路のコストを別々に設定出来る。
 */
#ifndef RAND
int map[TOWN_COUNT][TOWN_COUNT] =
{
	{  0,  0,  6, 10,  2, 10, 16, 17, 15, 15,  8,  6,  4, 18, 11, },
	{  0,  0, 19, 12,  0, 12,  1,  3,  7, 19,  1, 10,  5, 14,  9, },
	{  6, 19,  0, 12, 11, 19, 16,  1, 11, 15, 13, 14, 10, 19, 15, },
	{ 10, 12, 12,  0,  1, 12,  9,  8,  6,  4, 12,  3,  6,  4, 19, },
	{  2,  0, 11,  1,  0, 11,  7,  0, 15, 12,  8,  6,  7,  2, 14, },
	{ 10, 12, 19, 12, 11,  0,  9,  9,  3,  6,  1, 10, 18,  5, 19, },
	{ 16,  1, 16,  9,  7,  9,  0, 10, 13, 13,  9,  5,  2, 13, 17, },
	{ 17,  3,  1,  8,  0,  9, 10,  0, 16,  8,  0, 18, 16,  0,  2, },
	{ 15,  7, 11,  6, 15,  3, 13, 16,  0,  2,  7, 15, 16, 18, 16, },
	{ 15, 19, 15,  4, 12,  6, 13,  8,  2,  0, 19,  8, 19,  1, 17, },
	{  8,  1, 13, 12,  8,  1,  9,  0,  7, 19,  0, 15,  1, 15,  0, },
	{  6, 10, 14,  3,  6, 10,  5, 18, 15,  8, 15,  0, 10,  1, 17, },
	{  4,  5, 10,  6,  7, 18,  2, 16, 16, 19,  1, 10,  0, 15,  3, },
	{ 18, 14, 19,  4,  2,  5, 13,  0, 18,  1, 15,  1, 15,  0,  7, },
	{ 11,  9, 15, 19, 14, 19, 17,  2, 16, 17,  0, 17,  3,  7,  0, },
};
#else
int map[TOWN_COUNT][TOWN_COUNT];
#endif

/* individual：個体データ。個体は都市間を移動する経路(pathway)で表現され、
 *             その経路から算出されるcostで評価される。
 * 経路表現
 * 例えば、経路0-1-2-3-0の表現は
 *     1, 2, 3, 0
 * 経路0-3-1-2-0の表現は
 *     3, 2, 0, 1
 * となる。
*/
typedef struct
{
	int pathway[TOWN_COUNT];
	int cost;
} individual;

// population：対象となる集団の実体。
individual** population;

/*---------------------------------------------------------------------
 * 個体のcostの計算を行う。
 */
int calculate_cost(individual *target)
{
	int path_flag[TOWN_COUNT];
	int return_cost = 0;
	int i, j;

	// mapに従ってcostを加算。
	// 0の場合は接続がないとしてペナルティ分のcostを加算。
	for(i = 0; i < TOWN_COUNT; i++)
		if(map[i][target->pathway[i]] == 0)
			return_cost += PENALTY;
		else
			return_cost += map[i][target->pathway[i]];

	// 通過していない都市があった場合、ペナルティを科す。
	for (i = 0; i < TOWN_COUNT; i++) {
		path_flag[0] = 0;
		for(j = 0; j < TOWN_COUNT; j++)
			if(i == target->pathway[j])
				path_flag[0] = 1;
		if(path_flag[0] == 0)
			return_cost += PENALTY;
	}

	// 繋がっていない都市があった場合、ペナルティを科す。
	for(i = 0; i < TOWN_COUNT; i++)
		path_flag[i] = 1;
	path_flag[0] = 0;
	for(i = target->pathway[0]; path_flag[i] == 1; i = target->pathway[i])
		path_flag[i] = 0;
	for(i = 0, j = 0; i < TOWN_COUNT; i++)
		j += path_flag[i];
	return_cost += PENALTY * j;

	target->cost = return_cost;

	return return_cost;
}

/*---------------------------------------------------------------------
 * 個体の突然変異体を作る。
 */
void mutation(individual *target)
{
	// 染色体の1ヶ所をランダムに選んでランダムに変更。
	target->pathway[rand() % TOWN_COUNT] = rand() % TOWN_COUNT;
}

/*---------------------------------------------------------------------
 * 2つ個体から1点交差によって新しい個体を得る。
 */
void crossover(individual *target, individual *source1, individual *source2)
{
	int point;
	int i;

	// 交差位置を乱数で決定。
	point = rand() % (TOWN_COUNT - 1) + 1;

	// 染色体の前半をsource1からコピー。
	for(i = 0; i < point; i++)
		target->pathway[i] = source1->pathway[i];

	// 染色体の後半をsource2からコピー。
	for(; i < TOWN_COUNT; i++)
		target->pathway[i] = source2->pathway[i];
}

/*---------------------------------------------------------------------
 * GAに関する全ての操作の前に初期化を行う。
 */
void ga_init()
{
	int i, j;

	// 集団のメモリ領域を確保
	population = malloc(sizeof(individual *) * POPULATION_COUNT);
	for(i = 0; i < POPULATION_COUNT; i++)
		population[i] = malloc(sizeof(individual));

	// 集団の初期状態を生成する。
	for ( i = 0; i < POPULATION_COUNT; i++ )
		for ( j = 0; j < TOWN_COUNT; j++ )
			population[i]->pathway[j] = rand() % TOWN_COUNT;
}

/*---------------------------------------------------------------------
 * 各個体のcostを計算し順列に並べ替える。
 */
void ga_ranking()
{
	int i, j;
	individual *tmp;

	// 各個体のcostを計算。
	for(i = 0; i < POPULATION_COUNT; i++)
		calculate_cost(population[i]);

	// costに基づき並べ替える。
	// costのかかってない個体ほど上位にくる。
	for(i = 0; i < POPULATION_COUNT - 1; i++)
		for(j = i + 1; j < POPULATION_COUNT; j++)
			if(population[i]->cost > population[j]->cost) {
				tmp = population[i];
				population[i] = population[j];
				population[j] = tmp;
			}
}

/*---------------------------------------------------------------------
 * 次世代の集団を生成する。
 */
void ga_next_genelation()
{
	int i;

	// 集団は順位付けられているものとし、CUT_LINE以下の個体を
	// CUT_LINE以上の個体から1点交差によって生成される個体に置き換える。
	for(i = CUT_LINE; i < POPULATION_COUNT; i++)
		crossover(population[i], population[rand() % CUT_LINE],
			population[rand() % CUT_LINE]);

	// 全ての個体はMUTATION_ODDSの確率で突然変異をおこす。
	for(i = 0; i < POPULATION_COUNT; i++)
		if((rand() % (int)(1.0 / MUTATION_ODDS)) == 0)
			mutation(population[i]);
}

/*---------------------------------------------------------------------
 */
#ifdef GA_PROGRAM
int main()
{
	int i, j;
	time_t now_t, old_t;

	srand(0);

#ifdef RAND
	// 乱数によるmap自動生成
	for(i = 0; i < TOWN_COUNT - 1; i++)
		for(j = i + 1; j < TOWN_COUNT; j++) {
			map[i][j] = rand() % MAX_COST;
			map[j][i] = map[i][j];
		}
#endif

	ga_init();
	printf("\nmap\n");
	for(i = 0; i < TOWN_COUNT; i++) {
		for(j = 0; j < TOWN_COUNT; j++)
			printf(" %2d,", map[i][j]);
		printf("\n");
	}
	printf("\n");

	time(&old_t);

	for(i = 0; 1; ga_next_genelation(), i++) {
		ga_ranking();
		printf("%3d: cost %4d\n", i, population[0]->cost);
		if(population[0]->cost < STOP_COST)
			break;
	}
	printf("\n");

	time(&now_t);

	printf("path  0->");
	for(i = population[0]->pathway[0]; i != 0; i = population[0]->pathway[i])
		printf("%2d->", i);
	printf(" 0\n");

	for(i = 0; i < POPULATION_COUNT; i++)
		free(population[i]);
	free(population);

	printf("%ld sec\n", now_t - old_t);

	return 0;
}

/*---------------------------------------------------------------------
 * pathを検索して厳密解を求めるプログラムの例
 */
#else
int min_cost = STOP_COST;
int min_path[TOWN_COUNT + 1], min_path_point, true_min[TOWN_COUNT + 1];

void test_search(int now_town, int *no_path, int no_path_count, int now_cost)
{
	int new_cost, *new_no_path;
	int i, j;

	if(no_path_count == 1) {
		new_cost = now_cost + map[now_town][no_path[0]];
		if(map[now_town][no_path[0]] == 0)
			new_cost += PENALTY;
		if(new_cost < min_cost) {
			min_cost = new_cost;
			min_path[min_path_point] = no_path[0];
			memcpy(true_min, min_path, sizeof(int) * (TOWN_COUNT + 1));
		}
		return;
	}

	for(i = 0; i < no_path_count; i++) {
		new_cost = now_cost + map[now_town][no_path[i]];
		if(map[now_town][no_path[i]] == 0)
			new_cost += PENALTY;
		if(new_cost >= min_cost) {
			continue;
		}
		new_no_path = malloc(sizeof(int) * (no_path_count - 1));
		for(j = 0; j < no_path_count; j++) {
			if(i > j)
				new_no_path[j] = no_path[j];
			else if(i < j)
				new_no_path[j - 1] = no_path[j];
		}
		min_path[min_path_point] = no_path[i];
		min_path_point++;
		test_search(no_path[i], new_no_path, no_path_count - 1, new_cost);
		free(new_no_path);
		min_path_point--;
	}
}

int main()
{
	int no_path[TOWN_COUNT - 1];
	int i;
	time_t now_t, old_t;

	srand(0);

#ifdef RAND
	// 乱数によるmap自動生成
	for(i = 0; i < TOWN_COUNT - 1; i++)
		for(j = i + 1; j < TOWN_COUNT; j++) {
			map[i][j] = rand() % MAX_COST;
			map[j][i] = map[i][j];
		}
#endif

	time(&old_t);

	min_path[0] = 0;
	min_path_point = 1;

	for(i = 1; i < TOWN_COUNT; i++)
		no_path[i - 1] = i;

	test_search(0, no_path, TOWN_COUNT - 1, 0);

	time(&now_t);

	printf("cost %d\n", min_cost);
	printf("path ");
	for(i = 0; i < TOWN_COUNT; i++)
		printf("%2d->", true_min[i]);
	printf("%2d\n", true_min[i]);
	printf("%ld sec\n", now_t - old_t);

	return 0;
}
#endif
