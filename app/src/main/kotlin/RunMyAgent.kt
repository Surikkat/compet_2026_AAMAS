import games.planetwars.core.GameParams
import games.planetwars.runners.GameRunner
import games.planetwars.agents.random.BetterRandomAgent
import my.agents.MyAgent

fun main() {
    val params = GameParams(numPlanets = 20)
    val runner = GameRunner(MyAgent(), BetterRandomAgent(), params)
    val result = runner.runGame()
    println(result.statusString())
}