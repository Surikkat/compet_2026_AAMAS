package my.agents

import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.agents.Action
import games.planetwars.core.GameState

class MyAgent : PlanetWarsPlayer() {

    override fun getAction(state: GameState): Action {
        val myPlanet = state.planets
            .filter { it.owner == player && it.nShips > 1 }
            .randomOrNull() ?: return Action.doNothing()

        val target = state.planets
            .filter { it.owner != player }
            .randomOrNull() ?: return Action.doNothing()

        return Action(player, myPlanet.id, target.id, myPlanet.nShips / 2)
    }

    override fun getAgentType() = "My First Agent"
}