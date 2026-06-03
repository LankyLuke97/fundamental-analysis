import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import './InvestmentTrades.css'

const DATA_ENDPOINT = 'http://localhost:8000/transactions'; {/* Currently transactions is the only endpoint */ }
const FILTERS = { ticker: { value: '', type: 'text' }, dateFrom: { value: '', type: 'date' }, dateTo: { value: '', type: 'date' }, }

const transactionActions = {
    fetchInitial: 'FETCH_TRANSACTIONS',
    fetchSuccess: 'FETCH_SUCCESS',
    fetchFailure: 'FETCH_FAILURE',
};

const transactionReducer = (state, action) => {
    switch (action.type) {
        case transactionActions.fetchInitial:
            return {
                ...state,
                isLoading: true,
                isError: false,
            };
        case transactionActions.fetchSuccess:
            return {
                ...state,
                isLoading: false,
                isError: false,
                data: action.payload,
            };
        case transactionActions.fetchFailure:
            return {
                ...state,
                isLoading: false,
                isError: true,
            };
        default:
            throw new Error();
    }
};

const Transactions = () => {
    const [filters, setFilters] = useState(FILTERS);
    const [url, setUrl] = useState(DATA_ENDPOINT);
    const [transactionItems, dispatchTransactionItems] = useReducer(transactionReducer, {
        data: [],
        isLoading: false,
        isError: false,
    });

    const handleSearchInput = (key, value) =>
        setFilters({ ...filters, [key]: value });

    const handleSearchSubmit = event => {
        let queryString = new URLSearchParams(
            Object.entries(filters)
                .filter(([_, value]) => value != null && value.trim() != '')
        ).toString();

        queryString = queryString ? `?${queryString}` : '';
        setUrl(`${DATA_ENDPOINT}${queryString}`);

        event.preventDefault();
    };

    const fetchEndpoint = useCallback(async () => {
        dispatchTransactionItems({ type: transactionActions.fetchInitial });
        try {
            const response = await fetch(url);
            const result = await response.json();
            dispatchTransactionItems({
                payload: result,
                type: transactionActions.fetchSuccess,
            });
        } catch (error) {
            dispatchTransactionItems({ type: transactionActions.fetchFailure });
        }
    }, [url]);

    useEffect(() => { fetchEndpoint() }, [fetchEndpoint]);

    return (
        <>
            <SearchForm filters={filters} onSearchInput={handleSearchInput} onSearchSubmit={handleSearchSubmit} />
            <List items={transactionItems.data} />
        </>
    );
};

const SearchForm = ({ filters, onSearchInput, onSearchSubmit }) => (
    <form onSubmit={onSearchSubmit} className="search-form">
        {Object.entries(filters).map(([key, details]) =>
            <InputWithLabel key={key} id={`filter${key}`} details={details} isFocused onInputChange={(e) => onSearchInput(key, e.target.value)}>{key}:&nbsp;</InputWithLabel>
        )}
        <button className="button button_large" type="submit">&#x1F50D;</button>
    </form>
);

const InputWithLabel = ({ id, details, onInputChange, isFocused, children }) => {
    const inputRef = useRef();
    useEffect(() => {
        if (isFocused && inputRef.current) inputRef.current.focus();
    }, [isFocused]);

    return (
        <>
            <label className="label" htmlFor={id}>{children}</label>
            <input className="input" id={id} type={details.type} value={details.value} onChange={onInputChange} />
            {/* Always pass functions to handlers, not the return value -
                unless the function returns another function.
            */}
        </>
    );
};

const List = ({ items, children }) => (
    <>
        <ul>
            {items.map(item => <ListItem key={item.id} item={item} />)}
        </ul>
    </>
);

const ListItem = ({ item, children }) => (
    <>
        <li className="listBroad">
            <span style={{ width: '20%' }}>{item.ticker}</span>&nbsp;
            <span style={{ width: '15%' }}>{item.price}</span>&nbsp;
            <span style={{ width: '15%' }}>{item.quantity}</span>&nbsp;
            <span style={{ width: '50%' }}>{item.timestamp}</span>&nbsp;
        </li>
    </>
);

export default Transactions;
